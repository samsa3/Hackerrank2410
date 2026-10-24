"""Prompt and schema construction for the hybrid audit agent.

The model now makes a strictly bounded decision:
- authorized status context from deterministic code
- status decision
- relevant resources / missing evidence
- confidence and note

Deterministic code, not the model, constructs final steps. This avoids allowing
the model to generate invalid operation/state pairs or invent step formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CASE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["case_id", "deletion_status"],
    "properties": {
        "case_id": {"type": "string", "minLength": 1},
        "deletion_status": {
            "type": "string",
            "enum": ["COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"],
        },
        "relevant_resources": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning_summary": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You are an AI deletion-audit analyst.

Return one JSON object only. Use compact keys and no explanation outside JSON.

The caller has already supplied packet facts, linked resource IDs, and authorized_deletion. Treat authorized_deletion as authoritative.

Reliable statuses:
- COMPLETE: all relevant records received required treatment and no reason for concern at or before audit_at.
- INCOMPLETE: a relevant record, flow, or restore path remains in a prohibited state.
- NOT_APPLICABLE: non-deletion request (authorization decision supplied).

Rules:
- Do not re-invent resource IDs. Reuse only IDs from the packet.
- If no before-audit evidence is available for a linked resource, it is not automatically COMPLETE.
- Do not mark COMPLETE if any relevant safe resource is omitted from evidence.
- For authorization text, let the caller's authorized_deletion decide NOT_APPLICABLE.
- Use concise confidence between 0 and 1.
"""

AUDIT_DECISION_SCHEMA = CASE_SCHEMA


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
    deletion_status: Optional[str]
    relevant_resources: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_summary: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)


def render_packet_summary(packet: Any) -> str:
    lines = [
        f"case_id: {packet.case_id}",
        f"subject_email: {packet.email}",
        f"stable_user_id: {packet.user_id}",
        f"authorized_deletion: {packet.authorized_deletion}",
        f"requested_at: {packet.requested_at}",
        f"audit_at: {packet.audit_at}",
        f"identity_notes: {packet.identity_notes}",
        f"linked_resource_ids: {packet.linked_resource_ids}",
        "",
        "records:",
    ]

    for record in getattr(packet, "records", []):
        lines.append(
            f"- system={record.system_id}; resource={record.resource_id}; "
            f"category={record.data_category}; source={record.source_file_id}; "
            f"observed_at={record.observed_at}; fields={record.fields}"
        )

    omitted = getattr(packet, "omitted_records", [])
    if omitted:
        lines.append("")
        lines.append("omitted_evidence:")
        for item in omitted:
            lines.append(f"- {item}")

    return "\n".join(lines)


def build_user_prompt(packet: Any, allowed_evidence_ids: Optional[set[str]] = None) -> str:
    allowed = allowed_evidence_ids or set()
    return f"""
Apply the Ghost Data Still Exists deletion contract to the case packet.

Packet:
{render_packet_summary(packet)}

Allowed evidence IDs:
{sorted(allowed)}

Return JSON with only status, relevant resources, missing evidence, confidence, and a concise summary.
"""


def schema_text(schema: dict) -> str:
    return json.dumps(schema, sort_keys=True, indent=2)


# Optional commented Pydantic sketch intentionally retained because Pydantic may be
# added later without changing runtime interfaces.
#
# class CaseDecisionModel(BaseModel):
#     case_id: str
#     deletion_status: Literal["COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"]
#     relevant_resources: list[str] = []
#     missing_evidence: list[str] = []
#     confidence: float = 0.0
#     reasoning_summary: str = ""
