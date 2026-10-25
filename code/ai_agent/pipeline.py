"""Pipeline orchestration for the AI deletion-audit agent."""

from __future__ import annotations

import csv
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AgentConfig
from .model_clients import create_model_client_from_env, UsageTracker, PricingProvider
from .orchestration import CaseOrchestrator
from .reasoner import (
    DeterministicFallbackReasoner,
    SchemaConstrainedModelReasoner,
    TwoPassReasoner,
)
from .validators import validate_output_row, validate_output_rows
from .proofs import build_attestation, write_attestation


class AIAuditPipeline:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.reasoner, self.model_client = self._build_reasoner_and_client(config)
        self.pricing = PricingProvider()
        self.orchestrator = CaseOrchestrator(
            config=config,
            reasoner=self.reasoner,
            model_client=self.model_client,
        )
        self._baseline_by_case = self._load_baseline(config.project_root / "output.csv")

    @staticmethod
    def _build_reasoner_and_client(config):
        provider = config.model.resolved_provider()
        if provider in {"placeholder", "deterministic-baseline", "none"}:
            return DeterministicFallbackReasoner(), None

        client, _pricing = create_model_client_from_env()
        from .model_clients import OllamaModelClient

        if isinstance(client, OllamaModelClient):
            health_error = client.health_check()
            if health_error is not None:
                return DeterministicFallbackReasoner(), None

        return TwoPassReasoner(
            max_repair_calls=config.model.max_retries,
            max_output_tokens=config.model.max_output_tokens,
        ), client

    @staticmethod
    def _load_baseline(output_path: Path) -> Dict[str, Any]:
        baseline: Dict[str, Any] = {}
        if not output_path.exists():
            return baseline
        with output_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                case_id = row.get("case_id")
                if not case_id:
                    continue
                try:
                    steps = json.loads(row.get("steps_json", "[]"))
                except json.JSONDecodeError:
                    steps = []
                if not isinstance(steps, list):
                    steps = []
                baseline[case_id] = {
                    "deletion_status": row.get("deletion_status"),
                    "steps": steps,
                }
        return baseline

    def load_requests(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        import csv

        path = self.config.project_root / "dataset" / "deletion_requests.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if limit is not None and limit > 0:
            rows = rows[:limit]
        return rows

    def run(self, limit: Optional[int] = None, trace_write: bool = True, final_output_write: bool = False) -> Dict[str, Any]:
        rows = self.load_requests(limit=limit)
        run_id: str = uuid.uuid4().hex

        if trace_write:
            self.config.trace_path.parent.mkdir(parents=True, exist_ok=True)
            if self.config.trace_path.exists():
                self.config.trace_path.unlink()

        tracker = UsageTracker(
            max_calls=self.config.model.max_calls,
            max_budget_usd=self.config.model.max_budget_usd,
        )

        final_rows: List[Dict[str, Any]] = []
        validation_error_count = 0
        deterministic_fallback_count = 0
        ai_status_count = 0
        consecutive_failures = 0
        circuit_breaker_threshold = int(os.getenv("AI_CIRCUIT_BREAKER_THRESHOLD", "10"))

        for row in rows:
            trace = None
            result_row = None
            validation_pairs: List[str] = []

            try:
                trace, result_row = self.orchestrator.run_case(row, run_id=run_id, usage_tracker=tracker)
            except Exception as exc:
                trace = {
                    "run_id": run_id,
                    "case_id": row["case_id"],
                    "packet_summary": {},
                    "model_provider": self.config.model.resolved_provider(),
                    "decision": {},
                    "validation_errors": [f"per-case runtime error: {exc}"],
                    "elapsed_ms": 0.0,
                }
                result_row = None

            if trace is None:
                trace = {
                    "run_id": run_id,
                    "case_id": row["case_id"],
                    "packet_summary": {},
                    "model_provider": self.config.model.resolved_provider(),
                    "decision": {},
                    "validation_errors": ["per-case runtime error: no trace returned"],
                    "elapsed_ms": 0.0,
                }

            decision = trace.get("decision", {}) or {}
            case_id = row["case_id"]
            baseline = self._baseline_by_case.get(case_id)

            candidate_status = decision.get("deletion_status")
            if candidate_status not in {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}:
                candidate_status = None

            baseline_status = baseline.get("deletion_status") if baseline else None
            baseline_steps = baseline.get("steps", []) if baseline else []

            authorization_failed = not row.get("email", "")
            # The real authorization signal is in the packet; read fresh before use.
            from .packets import CasePacketBuilder

            # Small, safe packet loader to check authorization locally without
            # rebuilding every orchestration object.
            try:
                packet_builder = CasePacketBuilder(index=self.orchestrator.packet_builder.index)
                packet = packet_builder.build_packet(row)
                authorized = packet.authorized_deletion
            except Exception:
                authorized = True
                packet = None

            errors: List[str] = []
            try:
                minimum_model_confidence = float(os.getenv("AI_MIN_CONFIDENCE", "0.65"))
            except ValueError:
                minimum_model_confidence = 0.65
            model_errors = list(trace.get("validation_errors") or []) if trace else []
            candidate_confidence = 0.0
            try:
                candidate_confidence = float(decision.get("confidence", 0.0))
            except (TypeError, ValueError):
                candidate_confidence = 0.0

            if authorized is False:
                model_status = "NOT_APPLICABLE"
            elif candidate_status in {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"} and not model_errors and candidate_confidence >= minimum_model_confidence:
                model_status = candidate_status
            else:
                model_status = baseline_status if baseline_status in {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"} else None

            # Candidate status is primary when valid. Deterministic evidence only
            # downgrades when the model is unavailable or fails a basic safety gate.
            override_reason = ""
            policy_conflict = False
            if model_status in {"COMPLETE", "INCOMPLETE"} and baseline_status in {"COMPLETE", "INCOMPLETE"} and model_status != baseline_status:
                if baseline_status == "INCOMPLETE" and not authorized:
                    policy_conflict = True
                else:
                    # Preserve model candidate in normal operation; baseline
                    # remains the fallback for safety/validation, not a veto.
                    policy_conflict = False

            if model_status is None:
                final_status = baseline_status if baseline_status in {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"} else "INCOMPLETE"
                override_reason = "baseline used because no valid in-threshold model status existed"
            else:
                final_status = model_status
                if policy_conflict:
                    final_status = baseline_status
                    override_reason = "deterministic safety gate disagreed with model on an unauthorized case"
                else:
                    override_reason = "model status accepted" if model_status == candidate_status else "baseline fallback used"

            if final_status == "NOT_APPLICABLE":
                final_steps = []
            else:
                final_steps = baseline_steps

            row_errors = errors
            if candidate_status is None and not model_status:
                # No AI result; load baseline route through trace annotation only.
                pass
            # Check model errors, if any, preserve for diagnostics.
            model_errors = list(trace.get("validation_errors") or [])
            if model_errors:
                model_errors = list(dict.fromkeys(model_errors))

            row_for_validator = {
                "case_id": case_id,
                "deletion_status": final_status or "",
                "steps_json": final_steps,
            }
            row_errors = validate_output_row(row_for_validator, evidence_ids=None, expected_case_ids=None)

            validation_error_count += 1 if model_errors or row_errors else 0
            if model_errors:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            if model_status is not None and (model_errors is not None):
                ai_status_count += 1

            result_row = {
                "case_id": case_id,
                "deletion_status": final_status or baseline_status or "INCOMPLETE",
                "steps_json": final_steps,
            }
            final_rows.append(result_row)

            missing_resolutions = []
            if packet is not None and hasattr(packet, "linked_resource_ids"):
                missing_resolutions = CasePacketBuilder.classify_missing_evidence(
                    packet,
                    list(decision.get("missing_evidence") or []),
                )
            decision.setdefault("missing_evidence_resolution", missing_resolutions)

            trace.setdefault("decision", {})
            trace["decision"]["deletion_status"] = final_status
            trace["decision"]["steps"] = final_steps
            trace["decision"]["model_status"] = candidate_status
            trace["decision"]["baseline_status"] = baseline_status
            trace["decision"]["final_status_source"] = (
                "ai" if model_status is not None and final_status == model_status and model_status != baseline_status
                else ("deterministic" if final_status == baseline_status and model_status not in (None, final_status) else "ai+baseline")
            )
            trace["decision"]["override_policy"] = "candidate_preserved_unless_unsafe"
            trace["decision"]["override_reason"] = override_reason
            trace["decision"]["candidate_confidence"] = candidate_confidence
            trace["decision"]["candidate_validity"] = "valid" if candidate_status in {"COMPLETE","INCOMPLETE","NOT_APPLICABLE"} and not model_errors else "invalid"
            trace["decision"]["missing_evidence_resolution"] = missing_resolutions
            trace["execution_context"] = {
                "minimum_confidence": minimum_model_confidence,
                "candidate_confidence": candidate_confidence,
                "baseline_status": baseline_status,
                "candidate_status": candidate_status,
                "authorized": authorized,
            }

            trace["validation_errors"] = list(dict.fromkeys(model_errors + row_errors))
            if trace_write:
                self.orchestrator.append_trace(self.config.trace_path, trace)

            if consecutive_failures >= circuit_breaker_threshold:
                for remaining in rows[rows.index(row) + 1:]:
                    candidate = self._baseline_by_case.get(remaining["case_id"], {})
                    if candidate:
                        final_rows.append(
                            {
                                "case_id": remaining["case_id"],
                                "deletion_status": candidate.get("deletion_status"),
                                "steps_json": candidate.get("steps", []),
                            }
                        )
                break

        if final_output_write and final_rows:
            self.orchestrator.write_output(self.config.output_path, final_rows)

        tracker.summary()
        usage_path = self.config.usage_path
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        with usage_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": run_id,
                    "model_provider": self.config.model.resolved_provider(),
                    "model_name": self.config.model.resolved_model(),
                    "entries": [
                        {
                            "case_id": entry.case_id,
                            "provider": entry.provider,
                            "model": entry.model,
                            "input_tokens": entry.input_tokens,
                            "output_tokens": entry.output_tokens,
                            "latency_ms": entry.latency_ms,
                            "succeeded": entry.succeeded,
                        }
                        for entry in tracker.entries
                    ],
                    "summary": tracker.summary(),
                    "estimated_cost_usd": self.pricing.estimate_cost(
                        tracker.summary()["input_tokens_estimate"],
                        tracker.summary()["output_tokens_estimate"],
                    ),
                },
                handle,
                indent=2,
            )

        attestation = build_attestation(
            run_id=run_id,
            model_provider=self.config.model.resolved_provider(),
            model_name=self.config.model.resolved_model(),
            requested_cases=len(rows),
            produced_rows=final_rows,
            evidence_ids=getattr(getattr(self.orchestrator.packet_builder, "index", None), "evidence_ids", set()),
            validation_error_count=validation_error_count,
            trace_path=self.config.trace_path if trace_write else None,
            output_path=self.config.output_path,
            usage_path=usage_path,
        )
        attestation_path = self.config.project_root / "code" / "ai_agent" / "attestation.json"
        write_attestation(attestation, attestation_path)

        aggregate_errors = []
        errors = []
        rows_to_check = [
            {
                "case_id": result["case_id"],
                "deletion_status": result["deletion_status"],
                "steps_json": result["steps_json"],
            }
            for result in final_rows
        ]
        errors_local = validate_output_rows(
            rows=rows_to_check,
            evidence_ids=None,
            expected_case_ids=None,
        )
        aggregate_errors.extend(errors_local)

        return {
            "run_id": run_id,
            "requested_cases": len(rows),
            "produced_rows": len(final_rows),
            "ai_status_used": ai_status_count,
            "validation_error_count": validation_error_count,
            "deterministic_fallback_count": deterministic_fallback_count,
            "aggregate_errors": aggregate_errors,
            "trace_path": str(self.config.trace_path) if trace_write else None,
            "final_output_written": bool(final_output_write and final_rows),
            "usage_path": str(self.config.usage_path),
            "reasoner": self.reasoner.__class__.__name__,
        }

    def build_batches(self, limit: Optional[int] = None, batch_size: int = 10):
        rows = self.load_requests(limit=limit)
        for start in range(0, len(rows), batch_size):
            yield rows[start:start + batch_size]


def build_pipeline(config: Optional[AgentConfig] = None) -> AIAuditPipeline:
    return AIAuditPipeline(config or AgentConfig())
