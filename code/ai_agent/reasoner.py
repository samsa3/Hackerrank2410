"""Controlled AI-stage for the deletion-audit agent.

The model returns only:
- deletion status
- relevant resource IDs
- missing evidence
- confidence
- reasoning summary

All final row steps are built deterministically from packet and baseline data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from .model_clients import ModelClient, UsageTracker
from .packets import CasePacket
from .schema import AUDIT_DECISION_SCHEMA, SYSTEM_PROMPT, build_user_prompt, AuditDecision


class Reasoner:
    def reason(self, packet, allowed_evidence_ids, client=None, usage_tracker=None):
        raise NotImplementedError


class SchemaConstrainedModelReasoner(Reasoner):
    def __init__(self, max_repair_calls: int = 1, max_output_tokens: int = 1024) -> None:
        self.max_repair_calls = max(1, max_repair_calls)
        self.max_output_tokens = max(256, min(max_output_tokens, 4096))

    def reason(self, packet, allowed_evidence_ids, client=None, usage_tracker=None):
        if client is None:
            return AuditDecision(
                case_id=packet.case_id,
                deletion_status=None,
                relevant_resources=[],
                missing_evidence=["no model client available"],
                confidence=0.0,
                reasoning_summary="model stage unavailable",
            )

        if not getattr(packet, "authorized_deletion", True):
            return AuditDecision(
                case_id=packet.case_id,
                deletion_status="NOT_APPLICABLE",
                relevant_resources=[],
                missing_evidence=[],
                confidence=1.0,
                reasoning_summary="non-deletion request classified deterministically",
            )

        call = client.complete(
            case_id=packet.case_id,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(packet, allowed_evidence_ids),
            response_schema=AUDIT_DECISION_SCHEMA,
            options={"num_predict": self.max_output_tokens},
        )

        if usage_tracker is not None:
            from .model_clients import UsageEntry
            usage_tracker.record(
                UsageEntry(
                    case_id=packet.case_id,
                    provider=call.provider,
                    model=call.model,
                    input_tokens=call.input_tokens_estimate,
                    output_tokens=call.output_tokens_estimate,
                    latency_ms=call.latency_ms,
                    succeeded=call.error is None and isinstance(call.parsed_json, dict),
                )
            )

        if call.error is not None:
            return AuditDecision(
                case_id=packet.case_id,
                deletion_status=None,
                relevant_resources=[],
                missing_evidence=[call.error],
                confidence=0.0,
                reasoning_summary=call.error,
            )

        parsed = call.parsed_json if isinstance(call.parsed_json, dict) else {}
        resources = parsed.get("relevant_resources", [])
        if not isinstance(resources, list):
            resources = []
        missing = parsed.get("missing_evidence", [])
        if not isinstance(missing, list):
            missing = []
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        status = parsed.get("deletion_status")
        status = status if status in {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"} else None

        return AuditDecision(
            case_id=parsed.get("case_id", packet.case_id),
            deletion_status=status,
            relevant_resources=[str(x) for x in resources],
            missing_evidence=[str(x) for x in missing],
            confidence=min(1.0, max(0.0, confidence)),
            reasoning_summary=str(parsed.get("reasoning_summary", ""))[:500],
        )


class TwoPassReasoner(Reasoner):
    """Transparent two-pass reasoner: extraction first, deterministic policy pass second.

    The first pass is the existing schema-constrained model stage. The second pass
    does not call another model; it adds deterministic provenance and consistency
    checks around the first decision before returning it.
    """

    def __init__(self, max_repair_calls: int = 1, max_output_tokens: int = 1024) -> None:
        self.base = SchemaConstrainedModelReasoner(
            max_repair_calls=max_repair_calls,
            max_output_tokens=max_output_tokens,
        )

    @staticmethod
    def _policy_pass(packet, decision):
        from .schema import AuditDecision

        missing = list(decision.missing_evidence or [])
        if getattr(packet, "authorized_deletion", True):
            if decision.deletion_status == "NOT_APPLICABLE":
                missing.append("policy pass: authorized case cannot be NOT_APPLICABLE")
            if not getattr(packet, "records", []):
                missing.append("policy pass: no pre-audit evidence was available")

        return AuditDecision(
            case_id=decision.case_id,
            deletion_status=decision.deletion_status,
            relevant_resources=decision.relevant_resources,
            missing_evidence=list(dict.fromkeys(missing)),
            confidence=decision.confidence,
            reasoning_summary=(decision.reasoning_summary or "") + " | passes=extraction+deterministic-policy",
            steps=decision.steps,
        )

    def reason(self, packet, allowed_evidence_ids, client=None, usage_tracker=None):
        extraction_result = self.base.reason(
            packet=packet,
            allowed_evidence_ids=allowed_evidence_ids,
            client=client,
            usage_tracker=usage_tracker,
        )
        if client is None:
            return extraction_result
        return self._policy_pass(packet, extraction_result)


class DeterministicFallbackReasoner(Reasoner):
    def reason(self, packet, allowed_evidence_ids, client=None, usage_tracker=None):
        return AuditDecision(
            case_id=packet.case_id,
            deletion_status=None,
            relevant_resources=[],
            missing_evidence=["no reachable model endpoint"],
            confidence=0.0,
            reasoning_summary="deterministic fallback only",
        )
