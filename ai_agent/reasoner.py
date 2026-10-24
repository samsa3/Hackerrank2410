"""AI reasoner layer for the deletion-audit agent.

The model is bounded to one reasoning pass over a compact packet, plus exactly
one repair pass when schema validation or field parsing fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from ai_agent.model_clients import ModelClient, UsageTracker
from ai_agent.packets import CasePacket
from ai_agent.schema import AUDIT_DECISION_SCHEMA, SYSTEM_PROMPT, build_user_prompt, AuditDecision
from ai_agent.validators import validate_output_row


def _usage_entry(call: Any, case_id: str, tracker: Optional[UsageTracker]) -> None:
    if tracker is None:
        return
    usage_tracker_class = getattr(
        __import__("ai_agent.model_clients", fromlist=["UsageEntry"]),
        "UsageEntry",
    )
    tracker.record(
        usage_tracker_class(
            case_id=case_id,
            provider=call.provider,
            model=call.model,
            input_tokens=call.input_tokens_estimate,
            output_tokens=call.output_tokens_estimate,
            latency_ms=call.latency_ms,
            succeeded=call.error is None and call.parsed_json is not None,
        )
    )


def _decision_from_response(packet: CasePacket, response) -> AuditDecision:
    if response.error is not None:
        return AuditDecision(
            case_id=packet.case_id,
            deletion_status=None,
            steps=[],
            confidence=0.0,
            missing_evidence=[f"model call failed: {response.error}"],
            reasoning_summary=response.error,
        )

    parsed = response.parsed_json
    if not isinstance(parsed, dict):
        return AuditDecision(
            case_id=packet.case_id,
            deletion_status=None,
            steps=[],
            confidence=0.0,
            missing_evidence=["model response was not a JSON object"],
            reasoning_summary=str(response.content or "")[:1000],
        )

    return AuditDecision(
        case_id=parsed.get("case_id", packet.case_id),
        deletion_status=parsed.get("deletion_status"),
        steps=parsed.get("steps", []),
        missing_evidence=parsed.get("missing_evidence", []),
        confidence=parsed.get("confidence", 0.0),
        reasoning_summary=parsed.get("reasoning_summary", ""),
    )


class Reasoner:
    """Bounded one-stage reasoner interface."""

    def reason(self, packet: CasePacket, allowed_evidence_ids: Set[str], client=None, usage_tracker=None) -> AuditDecision:
        raise NotImplementedError


class SchemaConstrainedModelReasoner(Reasoner):
    """One model stage plus one bounded repair attempt."""

    def __init__(self, max_repair_calls: int = 1) -> None:
        self.max_repair_calls = max(1, max_repair_calls)

    @staticmethod
    def _validation_errors(decision: AuditDecision, allowed_evidence_ids: Set[str]) -> List[str]:
        row = {
            "case_id": decision.case_id,
            "deletion_status": decision.deletion_status or "",
            "steps_json": decision.steps,
        }
        return validate_output_row(row, evidence_ids=allowed_evidence_ids)

    def reason(self, packet: CasePacket, allowed_evidence_ids: Set[str], client=None, usage_tracker=None) -> AuditDecision:
        if client is None:
            return AuditDecision(
                case_id=packet.case_id,
                deletion_status=None,
                steps=[],
                confidence=0.0,
                missing_evidence=["No model client supplied"],
                reasoning_summary="Model stage skipped",
            )

        user_prompt = build_user_prompt(packet, allowed_evidence_ids)
        response = client.complete(
            case_id=packet.case_id,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=AUDIT_DECISION_SCHEMA,
        )
        _usage_entry(response, packet.case_id, usage_tracker)
        decision = _decision_from_response(packet, response)

        # If the first result fails basic validation, run exactly one bounded
        # repair pass. The repair repeats the schema and names the specific error.
        errors = self._validation_errors(decision, allowed_evidence_ids)
        if errors and self.max_repair_calls > 0 and response.error is None:
            repair_prompt = build_user_prompt(packet, allowed_evidence_ids) + f"""

Your previous answer had these validation errors:
- {chr(10).join(errors)}

Return corrected JSON that preserves every valid field and changes only what the
errors name.
"""
            repaired = client.complete(
                case_id=packet.case_id,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                response_schema=AUDIT_DECISION_SCHEMA,
            )
            _usage_entry(repaired, packet.case_id, usage_tracker)
            repaired_decision = _decision_from_response(packet, repaired)
            repaired_errors = self._validation_errors(repaired_decision, allowed_evidence_ids)
            if repaired_errors:
                errors = repaired_errors
            else:
                return repaired_decision

        if errors:
            decision.missing_evidence = list(dict.fromkeys(decision.missing_evidence + errors))
            decision.confidence = min(decision.confidence, 0.25)
        return decision


class DeterministicFallbackReasoner(Reasoner):
    """Called only when there is no reachable model endpoint."""

    def reason(self, packet: CasePacket, allowed_evidence_ids: Set[str], client=None, usage_tracker=None) -> AuditDecision:
        return AuditDecision(
            case_id=packet.case_id,
            deletion_status=None,
            steps=[],
            confidence=0.0,
            missing_evidence=["No reachable model endpoint was available"],
            reasoning_summary="Deterministic fallback only",
        )
