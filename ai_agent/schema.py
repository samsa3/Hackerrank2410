"""Prompt and schema construction for the hybrid AI audit agent.

The schema is duplicated intentionally as a plain Python dictionary: it doubles as
the schema sent to the model and as a documentation contract for maintainers.
Pydantic is optional and is therefore only sketched as comments below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from ai_agent.packets import CasePacket



STEP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "step_number_note": {"type": "string"},
    },
}


AUDIT_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["case_id", "deletion_status", "steps", "missing_evidence", "confidence"],
    "properties": {
        "case_id": {"type": "string", "minLength": 1},
        "deletion_status": {
            "type": "string",
            "enum": ["COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"],
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["phase", "system_id", "resource_id", "operation", "state", "evidence_id"],
                "properties": {
                    "phase": {"enum": ["FINDING", "REMEDIATION", "VERIFICATION"]},
                    "system_id": {"type": "string", "minLength": 1},
                    "resource_id": {"type": "string", "minLength": 1},
                    "operation": {"type": "string"},
                    "state": {"type": "string"},
                    "evidence_id": {"type": "string"},
                    "step_no": {"type": "integer", "minimum": 1},
                },
            },
        },
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning_summary": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You are an AI deletion-audit analyst.

The deterministic system has already determined authorized_deletion. Treat that value as authoritative and do not override it from evidence.
NOT_APPLICABLE means only that the request text did not authorize deletion.
A missing observation before the audit cutoff does not mean the resource is missing.
A resource in linked_resource_ids remains potentially relevant even if it has no observation at or before audit_at.
If evidence is missing or incomplete, add a missing_evidence entry and prefer INCOMPLETE over guessing COMPLETE or changing authorization.


Allowed phases and operations (exclusive):
Allowed state values: REPORT_REMAINING_RECORD->PRESENT; REPORT_EXPOSED_PII->PRESENT; REPORT_ACTIVE_DATA_FLOW->ACTIVE; REPORT_RESTORE_RISK->ACTIVE; DISABLE_DATA_FLOW->DISABLED; BLOCK_RESTORE->BLOCKED; DELETE_RECORD->ABSENT; REDACT_RECORD->REDACTED; ANONYMIZE_RECORD->ANONYMIZED; VERIFY_RECORD_ABSENT->ABSENT; VERIFY_RECORD_PRESENT->PRESENT; VERIFY_RECORD_REDACTED->REDACTED; VERIFY_RECORD_ANONYMIZED->ANONYMIZED; VERIFY_DATA_FLOW_DISABLED->DISABLED; VERIFY_RESTORE_BLOCKED->BLOCKED.

FINDING: REPORT_REMAINING_RECORD, REPORT_EXPOSED_PII, REPORT_ACTIVE_DATA_FLOW, REPORT_RESTORE_RISK
REMEDIATION: DISABLE_DATA_FLOW, BLOCK_RESTORE, DELETE_RECORD, REDACT_RECORD, ANONYMIZE_RECORD
VERIFICATION: VERIFY_RECORD_ABSENT, VERIFY_RECORD_PRESENT, VERIFY_RECORD_REDACTED, VERIFY_RECORD_ANONYMIZED, VERIFY_DATA_FLOW_DISABLED, VERIFY_RESTORE_BLOCKED

Use only valid (operation, state) pairs from the trial instructions.

Rules:
- Use only evidence supplied in the case packet.
- Never invent identifiers, evidence IDs, resources, or statuses.
- Apply the supplied deletion policy strictly.
- Do not mark COMPLETE if a finding remains.
- For NOT_APPLICABLE, output an empty steps array.
- For COMPLETE, findings and future remediation must be absent.
- For INCOMPLETE, include findings, remediation, and future verification in order.
- Return JSON only and match the supplied schema exactly.
"""


def render_packet_summary(packet: Any) -> str:
    """Render a copyable, bounded textual view of a packet for the model."""
    lines = [
        f"case_id: {packet.case_id}",
        f"subject_email: {packet.email}",
        f"stable_user_id: {packet.user_id}",
        f"authorized_deletion: {packet.authorized_deletion}",
        f"requested_at: {packet.requested_at}",
        f"audit_at: {packet.audit_at}",
        f"identity_notes: {packet.identity_notes}",
        "",
        "records:",
    ]

    for record in getattr(packet, "records", []):
        lines.append(
            f"- system={record.system_id}; resource={record.resource_id}; "
            f"category={record.data_category}; source={record.source_file_id}; observed_at={record.observed_at}; "
            f"fields={record.fields}"
        )

    omitted = getattr(packet, "omitted_records", [])
    if omitted:
        lines.append("")
        lines.append("omitted_evidence:")
        for item in omitted:
            lines.append(f"- {item}")

    linked = getattr(packet, "linked_resource_ids", [])
    if linked:
        lines.append("")
        lines.append(f"linked_resource_ids: {linked}")

    return "\n".join(lines)


def build_user_prompt(packet: Any, allowed_evidence_ids: set[str]) -> str:
    return f"""
Apply the Ghost Data Still Exists deletion contract to the following case packet.

Packet:
{render_packet_summary(packet)}

Allowed evidence IDs:
{sorted(allowed_evidence_ids)}

Response constraints:
- Use only operations, phases, and states from the contract.
- Findings must cite the evidence that proves them.
- If no applicable evidence exists, use an empty evidence_id and flag missing_evidence.
- Sort FINDING, then REMEDIATION, then VERIFICATION.
- Within each phase sort operation, then system_id, then resource_id.
- Number steps consecutively from 1.
- Return JSON matching the schema.
"""


@dataclass
class StepInput:
    phase: str
    system_id: str
    resource_id: str
    operation: str
    state: str
    evidence_id: str
    step_no: int = 0


@dataclass
class AuditDecision:
    case_id: str
    deletion_status: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "deletion_status": self.deletion_status,
            "steps": self.steps,
            "missing_evidence": self.missing_evidence,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
        }


def schema_text(schema: dict) -> str:
    return json.dumps(schema, sort_keys=True, indent=2)


# Optional commented Pydantic sketch. This is intentionally disabled because
# Pydantic is not currently installed in the project interpreter.
#
# class StepModel(BaseModel):
#     phase: Literal["FINDING", "REMEDIATION", "VERIFICATION"]
#     system_id: str
#     resource_id: str
#     operation: str
#     state: str
#     evidence_id: str
#     step_no: int = 0
#
# class AuditDecisionModel(BaseModel):
#     case_id: str
#     deletion_status: Literal["COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"]
#     steps: list[StepModel]
#     missing_evidence: list[str] = []
#     confidence: float = 0.0
#     reasoning_summary: str = ""
