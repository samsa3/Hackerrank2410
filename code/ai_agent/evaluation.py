"""Evaluation support for the AI audit agent.

The evaluator is deliberately deterministic and offline-friendly. It scores:
- deletion-status agreement
- exact JSON steps
- citation validity
- phase and operation order
- aggregate failure labels

The challenge also requires custom cases and a human-readable evaluation report.
Those files are expected under code/evaluation/ or ai_agent/evaluation/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import json


@dataclass
class MetricCounts:
    total: int = 0
    status_matches: int = 0
    exact_step_matches: int = 0
    citation_valid: int = 0
    schema_valid: int = 0


@dataclass
class EvaluationReport:
    model_provider: str
    case_count: int
    status_accuracy: float
    exact_step_accuracy: float
    citation_validity: float
    schema_validity: float
    mismatches: List[str] = field(default_factory=list)
    custom_cases: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class CustomCase:
    case_id: str
    email: str
    requested_at: str
    reported_status: str
    reported_at: str
    audit_at: str
    request_text: str
    expected_status: str
    expected_steps_json: str


def make_custom_cases() -> List[CustomCase]:
    """Three compact cases aligned to challenge risk patterns.

    TODO Step 8: Replace placeholders with real evidence-backed cases from a
    controlled fixture before final submission evaluation.
    """
    return [
        CustomCase(
            case_id="custom_access_request",
            email="lookup.only@example.com",
            requested_at="2026-08-01T09:00:00Z",
            reported_status="COMPLETE",
            reported_at="2026-08-02T09:00:00Z",
            audit_at="2026-08-03T09:00:00Z",
            request_text="Can you tell me what personal information you hold about me? This is an information request, not a request to erase it.",
            expected_status="NOT_APPLICABLE",
            expected_steps_json="[]",
        ),
        CustomCase(
            case_id="custom_multilingual_deletion",
            email="historical.alias@example.com",
            requested_at="2026-09-01T09:00:00Z",
            reported_status="PARTIALLY_COMPLETE",
            reported_at="2026-09-02T09:00:00Z",
            audit_at="2026-09-03T09:00:00Z",
            request_text="Solicito a exclusao permanente da conta associada a este e-mail.",
            expected_status="INCOMPLETE",
            expected_steps_json="[]",
        ),
        CustomCase(
            case_id="custom_restore_and_flow",
            email="restore.risk@example.com",
            requested_at="2026-09-05T09:00:00Z",
            reported_status="COMPLETE",
            reported_at="2026-09-06T09:00:00Z",
            audit_at="2026-09-07T09:00:00Z",
            request_text="I am asking you to remove my account and the personal information linked to it.",
            expected_status="INCOMPLETE",
            expected_steps_json="[]",
        ),
    ]


def _parse_json_safe(value: Any) -> Optional[Any]:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def citation_valid(steps: Any, allowed_evidence_ids: Set[str]) -> bool:
    parsed = _parse_json_safe(steps)
    if not isinstance(parsed, list):
        return False
    for step in parsed:
        if not isinstance(step, dict):
            return False
        evidence_id = step.get("evidence_id", "")
        if evidence_id not in allowed_evidence_ids and evidence_id != "":
            return False
    return True


def exact_step_list_match(actual: Any, expected: Any) -> bool:
    return _parse_json_safe(actual) == _parse_json_safe(expected)


def collect_metrics(
    actual_rows: List[Dict[str, str]],
    expected_rows: List[Dict[str, str]],
    model_provider: str,
    allowed_evidence_ids: Set[str],
    custom_cases: Optional[List[CustomCase]] = None,
) -> EvaluationReport:
    expected_by_case = {row["case_id"]: row for row in expected_rows}
    counts = MetricCounts()
    mismatches: List[str] = []

    for actual in actual_rows:
        case_id = actual["case_id"]
        expected = expected_by_case.get(case_id)
        counts.total += 1

        if expected is None:
            mismatches.append(f"{case_id}: no expected row")
            continue

        status_ok = actual.get("deletion_status") == expected.get("deletion_status")
        steps_ok = exact_step_list_match(actual.get("steps_json"), expected.get("steps_json"))
        citation_ok = citation_valid(actual.get("steps_json"), allowed_evidence_ids)
        structure_ok = isinstance(_parse_json_safe(actual.get("steps_json")), list)

        if status_ok:
            counts.status_matches += 1
        if steps_ok:
            counts.exact_step_matches += 1
        if citation_ok:
            counts.citation_valid += 1
        if structure_ok:
            counts.schema_valid += 1

        if status_ok and steps_ok and citation_ok:
            continue

        details = []
        if not status_ok:
            details.append(f"status {actual.get('deletion_status')} vs {expected.get('deletion_status')}")
        if not steps_ok:
            details.append("steps differ")
        if not citation_ok:
            details.append("citation invalid")
        details.extend(classify_failure(status_ok=status_ok, steps_ok=steps_ok, citation_ok=citation_ok))
        mismatches.append(f"{case_id}: " + "; ".join(details))

    denom = counts.total or 1
    return EvaluationReport(
        model_provider=model_provider,
        case_count=counts.total,
        status_accuracy=counts.status_matches / denom,
        exact_step_accuracy=counts.exact_step_matches / denom,
        citation_validity=counts.citation_valid / denom,
        schema_validity=counts.schema_valid / denom,
        mismatches=mismatches[:100],
        custom_cases=[{"case_id": c.case_id, "expected_status": c.expected_status} for c in (custom_cases or [])],
        notes=[
            "Evaluation is deterministic and does not use an LLM judge.",
            "Custom-case steps are placeholders pending evidence-backed expected values.",
        ],
    )


def write_report(report: EvaluationReport, path: Path) -> None:
    chunks = [
        "# Evaluation Report",
        "",
        f"- Model provider: {report.model_provider}",
        f"- Case count: {report.case_count}",
        f"- Status accuracy: {report.status_accuracy:.4f}",
        f"- Exact step accuracy: {report.exact_step_accuracy:.4f}",
        f"- Citation validity: {report.citation_validity:.4f}",
        f"- Schema validity (array): {report.schema_validity:.4f}",
        "",
        "## Mismatches",
    ]
    if report.mismatches:
        chunks.extend(f"- {item}" for item in report.mismatches)
    else:
        chunks.append("- none")
    chunks.extend(["", "## Custom Cases"])
    chunks.extend(f"- {case['case_id']}: expected {case['expected_status']}" for case in report.custom_cases)
    chunks.extend(["", "## Notes"])
    chunks.extend(f"- {note}" for note in report.notes)
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


# Failure-decomposition taxonomy, aligned to benchmark research on agent tool failures.
class FailureMode:
    STATUS_MISMATCH = "status_mismatch"
    STEP_MISMATCH = "step_mismatch"
    CITATION_MISMATCH = "citation_mismatch"
    STRUCTURE_MISMATCH = "structure_mismatch"
    AVAILABLE_MODEL_ONLY = "available_model_only"


def classify_failure(*, status_ok: bool, steps_ok: bool, citation_ok: bool, structure_ok: bool = True) -> List[str]:
    failures: List[str] = []
    if not status_ok:
        failures.append(FailureMode.STATUS_MISMATCH)
    if not steps_ok:
        failures.append(FailureMode.STEP_MISMATCH)
    if not citation_ok:
        failures.append(FailureMode.CITATION_MISMATCH)
    if not structure_ok:
        failures.append(FailureMode.STRUCTURE_MISMATCH)
    return failures
