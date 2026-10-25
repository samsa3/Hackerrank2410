"""Case packet assembly for the hybrid AI audit agent.

A case packet is the only information an LLM is allowed to reason over for one
case. It must be small, temporally clean, provenance-rich, and explicit about
what was omitted. Keeping this in a separate module keeps prompt and provider
code from accidentally reaching back into raw evidence.
"""

from __future__ import annotations

import importlib.util
import sys

from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass, field

from .indexes import DeterministicEvidenceIndex, IndexedRecord, parse_timestamp


@dataclass
class EvidenceRecord:
    system_id: str
    resource_id: str
    data_category: Optional[str]
    source_file_id: str
    observed_at: str
    fields: Dict[str, Any]


@dataclass
class CasePacket:
    case_id: str
    email: str
    user_id: Optional[str]
    authorized_deletion: bool
    requested_at: str
    audit_at: str
    records: List[EvidenceRecord] = field(default_factory=list)
    linked_resource_ids: List[str] = field(default_factory=list)
    omitted_records: List[Dict[str, str]] = field(default_factory=list)
    identity_notes: List[str] = field(default_factory=list)
    derivation_edges: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

# Load the existing deterministic helper module without needing `code/__init__.py`.
_DETERMINISTIC_PATH = PROJECT_ROOT / "code" / "deterministic_main.py"
_deterministic_spec = importlib.util.spec_from_file_location("deterministic_main", _DETERMINISTIC_PATH)
_deterministic = importlib.util.module_from_spec(_deterministic_spec)
_deterministic_spec.loader.exec_module(_deterministic)


class CasePacketBuilder:
    """Build one packet per deletion request, backed by deterministic evidence.

    TODO Step 6: add packet rendering into a prompt with strict schema instructions.
    TODO Step 8: connect these packets to exact expected forms for custom cases.
    """

    def __init__(self, index: Optional[DeterministicEvidenceIndex] = None) -> None:
        self.index = index or DeterministicEvidenceIndex()
        self._index_loaded = False

    def load_index(self) -> None:
        if self._index_loaded:
            return
        self.index.load()
        self._index_loaded = True

    def _authorized(self, request_text: str) -> bool:
        return bool(_deterministic.classify(request_text))

    def _identity_notes(self, user_id: Optional[str]) -> List[str]:
        if not user_id:
            return ["identity could not be resolved from the supplied email"]

        identity = self.index.identity_by_user.get(user_id)
        if identity is None:
            return [f"identity record for {user_id} is not available"]

        identifier_count = len(identity.identifiers)
        if identifier_count == 0:
            return [f"resolved {user_id}; no aliases recorded beyond stable user_id"]
        return [
            f"resolved stable identifiers for {user_id}",
            f"historical identifier count: {max(0, identifier_count - 1)}",
        ]

    @staticmethod
    def _to_evidence_record(record: IndexedRecord, user_id: Optional[str]) -> EvidenceRecord:
        # Keep fields compact and relevant to the model rather than raw record text.
        allowed = {
            "customer_reference", "user_id", "customer_email", "requester_email",
            "email", "full_name", "name", "phone", "email_hash", "status_code",
            "result", "status", "restore_allowed", "enabled", "record_type",
            "collection", "subject_reference", "retention_until", "message_excerpt",
            "object_key", "lookup_key",
        }
        compact = {}
        for key, value in (record.fields or {}).items():
            if key in allowed and value not in (None, "", "null", "none"):
                compact[key] = value

        return EvidenceRecord(
            system_id=record.system_id,
            resource_id=record.resource_id,
            data_category=record.data_category,
            source_file_id=record.file_id,
            observed_at=record.observed_at,
            fields=compact,
        )

    def build_packet(self, request_row: Dict[str, str]) -> CasePacket:
        """Build one temporally bounded, provenance-rich packet for one case."""
        self.load_index()

        case_id = request_row["case_id"]
        request_text = request_row["request_text"]
        requested_at = request_row["requested_at"]
        audit_at = request_row["audit_at"]
        audit_time = parse_timestamp(audit_at)

        authorized = self._authorized(request_text)
        user_id = _deterministic.resolve_user(request_row["email"]) if authorized else None

        mapped_resources = []
        if authorized and user_id is not None:
            mapped_resources = sorted(rid for rid, users in self.index.resource_to_users.items() if user_id in users)

        selected_records: List[IndexedRecord] = []
        omitted_records: List[Dict[str, str]] = []

        for resource_id in mapped_resources:
            history = self.index.resource_history(resource_id)
            if not history:
                continue

            on_or_before = [
                record for record in history if record.captured_dt <= audit_time
            ]
            after_audit = [
                record for record in history if record.captured_dt > audit_time
            ]

            if on_or_before:
                on_or_before.sort(key=lambda record: (record.captured_dt, record.file_id))
                selected_records.append(on_or_before[-1])
            else:
                latest = sorted(history, key=lambda record: (record.captured_dt, record.file_id))[-1]
                omitted_records.append(
                    {
                        "resource_id": resource_id,
                        "file_id": latest.file_id,
                        "captured_at": latest.captured_at,
                        "reason": "no evidence captured on or before audit_at; only later evidence exists",
                    }
                )

            for record in after_audit:
                omitted_records.append(
                    {
                        "resource_id": resource_id,
                        "file_id": record.file_id,
                        "captured_at": record.captured_at,
                        "reason": "captured after audit_at and therefore excluded from case state",
                    }
                )

        # Sort selected records in a deterministic secondary order for later prompts.
        selected_records.sort(key=lambda r: (r.system_id, r.resource_id))

        derivation_edges = self._build_derivation_edges(
            user_id=user_id,
            selected_records=selected_records,
            linked_resource_ids=mapped_resources,
            omitted_records=omitted_records,
        )

        return CasePacket(
            case_id=case_id,
            email=request_row["email"],
            user_id=user_id,
            authorized_deletion=authorized,
            requested_at=requested_at,
            audit_at=audit_at,
            records=[self._to_evidence_record(record, user_id) for record in selected_records],
            linked_resource_ids=mapped_resources,
            omitted_records=omitted_records,
            identity_notes=self._identity_notes(user_id),
            derivation_edges=derivation_edges,
            provenance=self._provenance_tags(selected_records=selected_records, identity_count=len(self._identity_notes(user_id))),
        )

    @staticmethod
    def _build_derivation_edges(
        *,
        user_id: Optional[str],
        selected_records: List["IndexedRecord"],
        linked_resource_ids: List[str],
        omitted_records: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Create deterministic transitive edges for resources and recurrence risk."""
        edges: List[Dict[str, Any]] = []
        for resource_id in sorted(linked_resource_ids):
            source = next((record.file_id for record in selected_records if record.resource_id == resource_id), None)
            edges.append(
                {
                    "from": "resolved_identity" if user_id else "unknown_identity",
                    "to": resource_id,
                    "edge_type": "resource_maps_to_user",
                    "source_file_id": source,
                    "audit_relevance": "in_scope",
                }
            )
        for omitted in omitted_records:
            edges.append(
                {
                    "from": omitted.get("resource_id"),
                    "to": omitted.get("file_id"),
                    "edge_type": "evidence_excluded_by_audit_at",
                    "reason": omitted.get("reason"),
                    "audit_relevance": "informational",
                }
            )
        for record in selected_records:
            lower_system = str(record.system_id).lower()
            lower_resource = str(record.resource_id).lower()
            lower_category = str(record.data_category or "").upper()
            if lower_system in ("integration_service", "export_service") or lower_resource.startswith(("flow_", "data_flow_")):
                edges.append(
                    {
                        "from": str(record.resource_id),
                        "to": record.system_id,
                        "edge_type": "recreation_risk_candidate",
                        "source_file_id": record.file_id,
                        "audit_relevance": "requires_active_flow_check",
                    }
                )
            if lower_category == "BACKUP_ARCHIVE":
                edges.append(
                    {
                        "from": record.resource_id,
                        "to": record.file_id,
                        "edge_type": "backup_retention_candidate",
                        "source_file_id": record.file_id,
                        "audit_relevance": "requires_restore_check",
                    }
                )
        return edges

    @staticmethod
    def _provenance_tags(*, selected_records: List["IndexedRecord"], identity_count: int) -> Dict[str, Any]:
        return {
            "origin": "DeterministicEvidenceIndex",
            "identity_sources": identity_count,
            "record_count": len(selected_records),
            "citation_rule": "file_id/activity_id/audit_id",
            "visibility": "deterministic_only",
        }

    def build_packets(self, rows: List[Dict[str, str]]) -> List[CasePacket]:
        return [self.build_packet(row) for row in rows]

    def deterministic_status(self, packet: CasePacket, index: DeterministicEvidenceIndex) -> str:
        if not packet.authorized_deletion:
            return "NOT_APPLICABLE"

        # Deterministic deletion status comes from evidence states, not model prose.
        linked_records = [
            (resource_id, self._best_record_for(index, resource_id, packet.records, packet.linked_resource_ids))
            for resource_id in packet.linked_resource_ids
        ]
        findings = False
        # If a mapped resource has no evidence at or before audit time, the case is
        # conservatively INCOMPLETE rather than assuming deletion succeeded.
        for resource_id, record in linked_records:
            if record is None:
                findings = True
                break
            noncompliant = self.is_record_noncompliant(resource_id, record)
            if noncompliant:
                findings = True
                break
        return "INCOMPLETE" if findings else "COMPLETE"

    @staticmethod
    def _best_record_for(index, resource_id, records, linked_ids):
        for record in records:
            if record.resource_id == resource_id:
                return record
        return None

    @staticmethod
    def packed_category(record: Optional[EvidenceRecord]) -> Optional[str]:
        return record.data_category if record is not None else None

    def is_record_noncompliant(self, resource_id: str, record: EvidenceRecord) -> bool:
        try:
            fields = record.fields or {}
            status_text = " ".join(str(v) for v in fields.values()).lower()
            direct = any(key in fields and fields[key] not in (None, "", "null", "none") for key in fields if key.lower() not in {"customer_reference", "user_id", "subject_reference"})

            category = record.data_category
            if category in ("SECURITY_AUDIT_LOG", "LEGAL_HOLD_RECORD"):
                return False

            if category == "BACKUP_ARCHIVE":
                try:
                    allowed = False
                    for key, value in fields.items():
                        if "restore_allowed" in key.lower():
                            allowed = str(value).lower() in {"1", "true", "yes"}
                    return allowed or "active" in status_text or "found" in status_text
                except Exception:
                    return False

            if category == "ACCOUNT_TOMBSTONE":
                return direct

            if category in ("ANALYTICS_PROFILE", "PRODUCT_TELEMETRY"):
                return direct

            if category == "EMAIL_DELIVERY_LOG":
                return direct

            if category == "SUPPORT_MESSAGE":
                return direct

            if category in ("BACKUP_ARCHIVE", "DATA_FLOW"):
                if "flow" in category or "backup" in status_text:
                    return "active" in status_text or "enabled" in status_text
                if "restore" in status_text or "backup" in status_text:
                    return "allowed" in status_text or "active" in status_text

            # Default: existing records without a deletion-compatible explicit null state.
            if isinstance(fields, dict):
                if any(value in ("null", "none", "", None, False) for value in fields.values()) and not direct:
                    return False
            return "present" in status_text or "found" in status_text or "active" in status_text or direct
        except Exception:
            return False

    def summary(self, packets: List[CasePacket]) -> Dict[str, Any]:
        authorized = sum(1 for p in packets if p.authorized_deletion)
        omitted = sum(len(p.omitted_records) for p in packets)
        return {
            "packet_count": len(packets),
            "authorized_packets": authorized,
            "not_applicable_packets": len(packets) - authorized,
            "omitted_evidence_records": omitted,
            "average_records_per_packet": (
                sum(len(p.records) for p in packets) / len(packets) if packets else 0.0
            ),
        }
