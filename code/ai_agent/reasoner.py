"""AI reasoning boundary for the governed audit agent.

The model returns only a compact case decision. Deterministic code constructs
 all contract-compliant steps, so the model cannot accidentally emit invalid
 operations, states, or JSON shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from .model_clients import ModelClient, UsageTracker
from .packets import CasePacket
from .schema import AUDIT_DECISION_SCHEMA, SYSTEM_PROMPT, build_user_prompt, StepInput, AuditDecision
from .validators import validate_output_row

ALLOWED_STATUSES = {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}


class Reasoner:
    def reason(self, packet, allowed_evidence_ids, client=None, usage_tracker=None):
        raise NotImplementedError


class SchemaConstrainedModelReasoner(Reasoner):
    def __init__(self, max_repair_calls: int = 1, max_output_tokens: int = 1024):
        self.max_repair_calls = 1
        self.max_output_tokens = max(256, min(max_output_tokens, 1024))

    @staticmethod
    def _coerce_decision(packet: CasePacket, decision: Any) -> AuditDecision:
        if not isinstance(decision, dict):
            decision = {}

        status = decision.get("deletion_status")
        status = status if status in ALLOWED_STATUSES else None
        resources = decision.get("relevant_resources", [])
        if not isinstance(resources, list):
            resources = []

        missing = decision.get("missing_evidence", [])
        if not isinstance(missing, list):
            missing = []

        try:
            confidence = float(decision.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        return AuditDecision(
            case_id=decision.get("case_id", packet.case_id),
            deletion_status=status,
            relevant_resources=[str(item) for item in resources],
            missing_evidence=[str(item) for item in missing],
            confidence=confidence,
            reasoning_summary=str(decision.get("reasoning_summary", ""))[:500],
            steps=[],
        )

    def _call(self, client, packet, allowed, prompt):
        return client.complete(
            case_id=packet.case_id,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            response_schema=AUDIT_DECISION_SCHEMA,
            options={"num_predict": self.max_output_tokens},
        )

    def reason(self, packet, allowed_evidence_ids, client=None, usage_tracker=None):
        if client is None:
            return AuditDecision(
                case_id=packet.case_id,
                deletion_status=None,
                relevant_resources=[],
                missing_evidence=["no model client"],
                confidence=0.0,
                reasoning_summary="model stage skipped",
                steps=[],
            )

        if not getattr(packet, "authorized_deletion", True):
            return AuditDecision(
                case_id=packet.case_id,
                deletion_status="NOT_APPLICABLE",
                relevant_resources=[],
                missing_evidence=[],
                confidence=1.0,
                reasoning_summary="deterministic authorization override",
                steps=[],
            )

        prompt = build_user_prompt(packet, allowed_evidence_ids)
        call = self._call(client, packet, allowed_evidence_ids, prompt)

        from .model_clients import UsageEntry
        if usage_tracker is not None:
            usage_tracker.record(
                UsageEntry(
                    case_id=packet.case_id,
                    provider=call.provider,
                    model=call.model,
                    input_tokens=call.input_tokens_estimate,
                    output_tokens=call.output_tokens_estimate,
                    latency_ms=call.latency_ms,
                    succeeded=call.error is None and call.parsed_json is not None,
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
                steps=[],
            )

        if not isinstance(call.parsed_json, dict):
            return AuditDecision(
                case_id=packet.case_id,
                deletion_status=None,
                relevant_resources=[],
                missing_evidence=["model response was not a JSON object"],
                confidence=0.0,
                reasoning_summary=str(call.content)[:250],
                steps=[],
            )

        return self._coerce_decision(packet, call.parsed_json)


class DeterministicFallbackReasoner(Reasoner):
    def reason(self, packet, allowed_evidence_ids, client=None, usage_tracker=None):
        return AuditDecision(
            case_id=packet.case_id,
            deletion_status=None,
            relevant_resources=[],
            missing_evidence=["no reachable model endpoint"],
            confidence=0.0,
            reasoning_summary="deterministic fallback only",
            steps=[],
        )
