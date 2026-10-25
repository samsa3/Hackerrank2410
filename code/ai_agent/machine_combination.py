"""Machine-assisted combination of model candidates and deterministic evidence.

The LLM provides a candidate deletion status. This module makes the actual
combination policy explicit and testable: a candidate contributes to the final
result only after deterministic packet and evidence constraints are satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


class CandidateMachineVerifier:
    """A deterministic verifier for schema-valid, packet-local candidates."""

    def evaluate(
        self,
        *,
        packet: Any,
        candidate_status: Optional[str],
        candidate_resources: List[str],
        missing_evidence: List[str],
        confidence: float,
        baseline_status: Optional[str],
        model_errors: List[str],
        minimum_confidence: float,
        allow_disagreement: bool = False,
    ) -> "CandidateVerification":
        reasons: List[str] = []

        if candidate_status not in {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}:
            reasons.append("candidate_status_is_not_schema_valid")
        if model_errors:
            reasons.append("model_errors_present")
        if confidence < minimum_confidence:
            reasons.append("confidence_below_minimum_gate")

        linked = set(getattr(packet, "linked_resource_ids", []) or [])
        if not isinstance(candidate_resources, list):
            reasons.append("relevant_resources_is_not_a_list")
        else:
            foreign = [r for r in candidate_resources if r not in linked]
            if foreign:
                reasons.append("candidate_cites_unknown_resources")

        if candidate_status == "COMPLETE":
            if missing_evidence:
                reasons.append("candidate_claims_complete_but_lists_remaining_evidence")
            reasoning = getattr(packet.packet if hasattr(packet, "packet") else None, "reasoning_summary", "")
            if any(word in str(reasoning).lower() for word in ("incomplete", "not complete", "remaining", "unresolved")):
                reasons.append("candidate_claims_complete_but_explanation_reports_unresolved_data")

        elif candidate_status == "INCOMPLETE":
            if not missing_evidence and not candidate_resources:
                reasons.append("candidate_claims_incomplete_without_unresolved_evidence_or_resources")
            if any(item for item in missing_evidence if not isinstance(item, str) or not item.strip()):
                reasons.append("missing_evidence_contains_non_string_or_empty_entry")
        elif candidate_status == "NOT_APPLICABLE":
            if getattr(packet, "authorized_deletion", True):
                reasons.append("candidate_claims_not_applicable_for_authorized_deletion")

        if baseline_status in {"COMPLETE", "INCOMPLETE"} and candidate_status not in {None, "NOT_APPLICABLE"}:
            if candidate_status != baseline_status and not allow_disagreement:
                reasons.append("candidate_does_not_agree_with_deterministic_baseline")

        return CandidateVerification(passed=not reasons, reasons=reasons)


@dataclass
class CandidateVerification:
    passed: bool
    reasons: List[str]


def combine_candidate(
    *,
    packet: Any,
    candidate_status: Optional[str],
    candidate_resources: List[Any],
    missing_evidence: List[Any],
    confidence: float,
    baseline_status: Optional[str],
    model_errors: List[str],
    minimum_confidence: float,
    allow_disagreement: bool = False,
) -> CandidateVerification:
    verifier = CandidateMachineVerifier()
    return verifier.evaluate(
        packet=packet,
        candidate_status=candidate_status,
        candidate_resources=[str(item) for item in (candidate_resources or [])],
        missing_evidence=[str(item) for item in (missing_evidence or [])],
        confidence=confidence,
        baseline_status=baseline_status,
        model_errors=model_errors,
        minimum_confidence=minimum_confidence,
        allow_disagreement=allow_disagreement,
    )
