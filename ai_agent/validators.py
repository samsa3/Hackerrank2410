"""Deterministic validation layer for the AI audit agent.

These validators are intentionally code-based rather than prompt-only. The model
may produce a plausible answer, but the output must cross this boundary before
it can be written to output.csv.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Set

ALLOWED_STATUSES = {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}
ALLOWED_PHASES = {"FINDING", "REMEDIATION", "VERIFICATION"}
PHASE_ORDER = {"FINDING": 0, "REMEDIATION": 1, "VERIFICATION": 2}
OPERATION_ORDER = {
    "FINDING": [
        "REPORT_REMAINING_RECORD",
        "REPORT_EXPOSED_PII",
        "REPORT_ACTIVE_DATA_FLOW",
        "REPORT_RESTORE_RISK",
    ],
    "REMEDIATION": [
        "DISABLE_DATA_FLOW",
        "BLOCK_RESTORE",
        "DELETE_RECORD",
        "REDACT_RECORD",
        "ANONYMIZE_RECORD",
    ],
    "VERIFICATION": [
        "VERIFY_RECORD_ABSENT",
        "VERIFY_RECORD_PRESENT",
        "VERIFY_RECORD_REDACTED",
        "VERIFY_RECORD_ANONYMIZED",
        "VERIFY_DATA_FLOW_DISABLED",
        "VERIFY_RESTORE_BLOCKED",
    ],
}
VALID_PAIRSTATES = {
    ("REPORT_REMAINING_RECORD", "PRESENT"),
    ("REPORT_EXPOSED_PII", "PRESENT"),
    ("REPORT_ACTIVE_DATA_FLOW", "ACTIVE"),
    ("REPORT_RESTORE_RISK", "ACTIVE"),
    ("DISABLE_DATA_FLOW", "DISABLED"),
    ("BLOCK_RESTORE", "BLOCKED"),
    ("DELETE_RECORD", "ABSENT"),
    ("REDACT_RECORD", "REDACTED"),
    ("ANONYMIZE_RECORD", "ANONYMIZED"),
    ("VERIFY_RECORD_ABSENT", "ABSENT"),
    ("VERIFY_RECORD_PRESENT", "PRESENT"),
    ("VERIFY_RECORD_REDACTED", "REDACTED"),
    ("VERIFY_RECORD_ANONYMIZED", "ANONYMIZED"),
    ("VERIFY_DATA_FLOW_DISABLED", "DISABLED"),
    ("VERIFY_RESTORE_BLOCKED", "BLOCKED"),
}
REQUIRED_STEP_KEYS = {"step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id"}


def validate_step(step: Any, index: int, evidence_ids: Optional[Set[str]] = None) -> List[str]:
    """Validate one step dictionary and return all found issues."""
    errors: List[str] = []

    if not isinstance(step, dict):
        return [f"step[{index}] is not a dict"]

    missing = REQUIRED_STEP_KEYS - set(step.keys())
    if missing:
        errors.append(f"step[{index}] missing keys: {sorted(missing)}")

    phase = step.get("phase")
    if phase not in ALLOWED_PHASES:
        errors.append(f"step[{index}] invalid phase: {phase!r}")

    operation = step.get("operation")
    state = step.get("state")
    if (operation, state) not in VALID_PAIRSTATES:
        errors.append(f"step[{index}] invalid operation/state pairing: {operation!r}/{state!r}")

    style = step.get("evidence_id", None)
    if not isinstance(style, str):
        errors.append(f"step[{index}] evidence_id must be a string")

    if evidence_ids is not None and style not in evidence_ids and style != "":
        errors.append(f"step[{index}] evidence_id not in evidence manifest: {style!r}")

    # Audit author should be either a valid ID or a documented gap.
    return errors


def validate_and_fix_step_order(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return steps in challenge-required order, retaining objects."""

    def sort_key(step: Dict[str, Any]) -> tuple:
        phase = step.get("phase", "ZZZ")
        operation = step.get("operation", "ZZZ")
        op_order = OPERATION_ORDER.get(phase, []).index(operation) if operation in OPERATION_ORDER.get(phase, []) else 999
        return (
            PHASE_ORDER.get(phase, 999),
            op_order,
            str(step.get("system_id", "")),
            str(step.get("resource_id", "")),
        )

    return sorted(steps, key=sort_key)


def validate_steps_json(value: Any, evidence_ids: Optional[Set[str]] = None) -> List[str]:
    errors: List[str] = []

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            return [f"steps_json is not valid JSON: {exc}"]

    if not isinstance(value, list):
        return [f"steps_json must be an array, got {type(value).__name__}"]

    # Step number check.
    for index, step in enumerate(value, start=1):
        if isinstance(step, dict) and "step_no" in step:
            if step.get("step_no") != index:
                errors.append(f"step_no must be consecutive from 1, found {step.get('step_no')!r} at index {index}")
        errors.extend(validate_step(step, index, evidence_ids))

    # Phase and operation-specific cross-rule checks.
    phases = [step.get("phase") if isinstance(step, dict) else None for step in value]
    seen_phase_order = []
    for phase in phases:
        if phase not in seen_phase_order:
            seen_phase_order.append(phase)

    # Ensure phase ordering appears as FINDING then REMEDIATION then VERIFICATION when present.
    expected_sequence = [p for p in ["FINDING", "REMEDIATION", "VERIFICATION"] if p in seen_phase_order]
    if seen_phase_order != expected_sequence:
        errors.append(f"phase order must be FINDING, REMEDIATION, VERIFICATION; got {seen_phase_order!r}")

    return errors


def validate_output_row(row: Any, evidence_ids: Optional[Set[str]] = None, expected_case_ids: Optional[Set[str]] = None) -> List[str]:
    errors: List[str] = []

    if not isinstance(row, dict):
        return ["row is not a dict"]

    for key in ("case_id", "deletion_status", "steps_json"):
        if key not in row:
            errors.append(f"missing column: {key}")

    status = row.get("deletion_status")
    if status not in ALLOWED_STATUSES:
        errors.append(f"invalid deletion_status: {status!r}")

    case_id = row.get("case_id")
    if expected_case_ids is not None and case_id not in expected_case_ids:
        errors.append(f"unknown case_id: {case_id!r}")

    steps = row.get("steps_json")
    errors.extend(validate_steps_json(steps, evidence_ids))

    if status == "NOT_APPLICABLE":
        try:
            parsed = json.loads(steps) if isinstance(steps, str) else steps
        except Exception:
            parsed = steps
        if parsed != []:
            errors.append("NOT_APPLICABLE must have an empty steps_json array")

    return errors


def validate_output_rows(rows: List[Any], evidence_ids: Optional[Set[str]] = None, expected_case_ids: Optional[Set[str]] = None) -> List[str]:
    errors: List[str] = []

    if not isinstance(rows, list):
        return ["rows must be a list"]

    if expected_case_ids is not None:
        seen = [r.get("case_id") if isinstance(r, dict) else None for r in rows]
        if len(seen) != len(set(seen)):
            errors.append("duplicate case_id in output")
        if set(seen) != expected_case_ids:
            errors.append(f"output case set differs from expected; missing={expected_case_ids - set(seen)}, extra={set(seen) - expected_case_ids}")
        if seen != sorted(expected_case_ids):
            errors.append("output rows must be sorted by case_id")

    for index, row in enumerate(rows):
        errors.extend(validate_output_row(row, evidence_ids, expected_case_ids))

    return errors


def model_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def raise_for_validation(errors: List[str]) -> None:
    if errors:
        raise ValueError("; ".join(errors))
