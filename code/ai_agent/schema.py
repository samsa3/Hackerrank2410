"""Prompt and schema construction for the governed AI audit agent.

The AI stage returns only bounded audit results:
- deletion status
- relevant resource IDs
- missing evidence
- confidence
- reasoning summary

All final steps are constructed by deterministic code, so the model can never
produce an invalid operation, state, phase, or step serialization.
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

AUDIT_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["case_id", "deletion_status", "relevant_resources", "missing_evidence", "confidence", "reasoning_summary"],
    "properties": {
        "case_id": {"type": "string", "minLength": 1},
        "deletion_status": {"type": "string", "enum": ["COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"]},
        "relevant_resources": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning_summary": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You are an AI deletion-audit analyst.

Return exactly one JSON object. No markdown, no commentary, no extra keys.

The deterministic backend has already resolved identity and candidate evidence.
Treat authorized_deletion as authoritative. NOT_APPLICABLE usually cannot be
derived unless the request does not request deletion.

Rules:
- Reuse only resource IDs from linked_resource_ids.
- If no before-audit evidence is available for a linked resource, treat it as potentially relevant and not automatically COMPLETE.
- If any relevant record still exists or a prohibited flow remains active, choose INCOMPLETE.
- Consider retained records as allowed when their required treatment is satisfied.
- Missing evidence must be listed as concrete IDs or exact gaps.
- Do not invent step objects, operations, states, or citation IDs.

Examples:
A COMPLETE packet necessarily has no policy-violating record or active risk at or before audit_at.
An INCOMPLETE packet contains at least one unresolved record, active data flow, expose PII, or restore risk.

Output JSON only.
"""

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
    compact = []
    compact.append(f"case_id={packet.case_id}")
    compact.append(f"email={packet.email}")
    compact.append(f"user_id={packet.user_id}")
    compact.append(f"authorized={packet.authorized_deletion}")
    compact.append(f"requested_at={packet.requested_at}")
    compact.append(f"audit_at={packet.audit_at}")
    compact.append(f"linked_resources={packet.linked_resource_ids}")
    compact.append(f"identity_notes={packet.identity_notes}")

    packet_records = getattr(packet, "records", [])
    if packet_records:
        compact.append("records=")
        for record in packet_records:
            compact.append(
                f"  system={record.system_id}, resource={record.resource_id}, "
                f"category={record.data_category}, source={record.source_file_id}, "
                f"observed_at={record.observed_at}, fields={record.fields}"
            )
    else:
        compact.append("records=none before audit_at")

    return "\n".join(compact)


def build_user_prompt(packet: Any, allowed_evidence_ids: Optional[Set[str]] = None) -> str:
    allowed = allowed_evidence_ids or set()
    return f"""Apply the Ghost Data Still Exists deletion contract to the compact case packet.

{render_packet_summary(packet)}

allowed_evidence_ids={sorted(allowed)}

Return only JSON:
case_id, deletion_status, relevant_resources, missing_evidence, confidence, reasoning_summary
"""


def schema_text(schema: dict) -> str:
    return json.dumps(schema, sort_keys=True, indent=2)


# Optional commented Pydantic sketch. Disabled because Pydantic is not installed.
#
# class AuditDecisionModel(BaseModel):
#     case_id: str
#     deletion_status: Literal["COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"]
#     relevant_resources: list[str] = []
#     missing_evidence: list[str] = []
#     confidence: float = 0.0
#     reasoning_summary: str = ""
