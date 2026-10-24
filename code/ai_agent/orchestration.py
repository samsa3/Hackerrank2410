"""End-to-end orchestration glue for one case at a time."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .validators import validate_output_row


class CaseOrchestrator:
    """Owns one case run end to end."""

    def __init__(self, config, reasoner, model_client=None) -> None:
        self.config = config
        self.index = None
        from .indexes import DeterministicEvidenceIndex
        self.index = DeterministicEvidenceIndex(config.project_root)
        self.index.load()
        from .packets import CasePacketBuilder
        self.packet_builder = CasePacketBuilder(self.index)
        self.reasoner = reasoner
        self.model_client = model_client

    @staticmethod
    def iso_utc(now=None):
        if now is None:
            now = datetime.now(timezone.utc)
        return now.isoformat().replace("+00:00", "Z")

    def allowed_evidence_ids(self, packet) -> set[str]:
        ids = {record.source_file_id for record in packet.records}
        ids.update(
            item.get("file_id", "")
            for item in packet.omitted_records
            if isinstance(item, dict) and item.get("file_id")
        )
        return {item for item in ids if item}

    def run_case(self, row, run_id=None, usage_tracker=None):
        run_id = run_id or uuid.uuid4().hex
        started = datetime.now(timezone.utc)
        packet = self.packet_builder.build_packet(row)

        from ai_agent.schema import AuditDecision

        original_model_status = None
        if getattr(packet, "authorized_deletion", True):
            decision = self.reasoner.reason(
                packet=packet,
                allowed_evidence_ids=self.allowed_evidence_ids(packet),
                client=self.model_client,
                usage_tracker=usage_tracker,
            )
            original_model_status = decision.deletion_status
            # Deterministic evidence status takes precedence because model output can
            # confuse "existing" records with successful deletion. Keep AI result only
            # for trace diagnostics and missing evidence context.
            decision.deletion_status = self.packet_builder.deterministic_status(packet, self.index)
            decision.missing_evidence = list(dict.fromkeys((decision.missing_evidence or []) + (decision.missing_evidence or [])))
            if original_model_status is not None:
                decision.reasoning_summary = (
                    (decision.reasoning_summary or "") + " | model_status=" + str(original_model_status)
                ).strip(" |")
        else:
            from ai_agent.schema import AuditDecision

            decision = AuditDecision(
                case_id=packet.case_id,
                deletion_status="NOT_APPLICABLE",
                steps=[],
                confidence=1.0,
                missing_evidence=[],
                reasoning_summary="deterministic request authorization override",
            )

        normalized_steps = []
        for index, step in enumerate(decision.steps or [], start=1):
            if not isinstance(step, dict):
                continue
            normalized = {key: value for key, value in step.items()}
            normalized["step_no"] = normalized.get("step_no", index)
            normalized_steps.append(normalized)
        decision.steps = normalized_steps

        errors = []
        row_for_validation = {
            "case_id": decision.case_id,
            "deletion_status": decision.deletion_status or "",
            "steps_json": normalized_steps,
        }
        errors.extend(
            validate_output_row(row_for_validation, evidence_ids=self.allowed_evidence_ids(packet))
        )

        result_row = None
        if not errors and decision.deletion_status == "NOT_APPLICABLE":
            result_row = {
                "case_id": decision.case_id,
                "deletion_status": "NOT_APPLICABLE",
                "steps_json": [],
            }
        elif not errors and decision.deletion_status is not None:
            result_row = {
                "case_id": decision.case_id,
                "deletion_status": decision.deletion_status,
                "steps_json": normalized_steps,
            }

        trace = {
            "run_id": run_id,
            "case_id": row["case_id"],
            "started_at": self.iso_utc(started),
            "packet_summary": {
                "authorized": packet.authorized_deletion,
                "user_id": packet.user_id,
                "record_count": len(packet.records),
                "omitted_record_count": len(packet.omitted_records),
                "linked_resource_count": len(packet.linked_resource_ids),
            },
            "model_provider": 
                (self.model_client.provider if getattr(self.model_client, "provider", None)
                 else None) or getattr(self.reasoner, "model_provider", None) or self.config.model.resolved_provider(),
            "decision": {
                "case_id": decision.case_id,
                "deletion_status": decision.deletion_status,
                "steps": decision.steps,
                "missing_evidence": decision.missing_evidence,
                "confidence": decision.confidence,
                "reasoning_summary": getattr(decision, "reasoning_summary", ""),
            },
            "validation_errors": errors,
            "elapsed_ms": (datetime.now(timezone.utc) - started).total_seconds() * 1000.0,
            "finished_at": self.iso_utc(datetime.now(timezone.utc)),
        }

        return trace, result_row

    def append_trace(self, path: Path, trace: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, separators=(",", ":")) + "\n")

    def write_output(self, path: Path, rows: List[dict]) -> None:
        import csv

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["case_id", "deletion_status", "steps_json"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "case_id": row["case_id"],
                        "deletion_status": row["deletion_status"],
                        "steps_json": json.dumps(row["steps_json"], separators=(",", ":")),
                    }
                )
