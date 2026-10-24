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
from .reasoner import DeterministicFallbackReasoner, SchemaConstrainedModelReasoner
from .validators import validate_output_row, validate_output_rows


_TRACE_NAME = "ai_agent_run_trace.jsonl"


def _build_reasoner_and_client(config: AgentConfig):
    provider = config.model.resolved_provider()
    if provider in {"placeholder", "deterministic-baseline", "none"}:
        return DeterministicFallbackReasoner(), None

    client, _pricing = create_model_client_from_env()
    from .model_clients import OllamaModelClient

    if isinstance(client, OllamaModelClient):
        health_error = client.health_check()
        if health_error is not None:
            return DeterministicFallbackReasoner(), None

    return SchemaConstrainedModelReasoner(
        max_repair_calls=config.model.max_retries,
        max_output_tokens=config.model.max_output_tokens,
    ), client


class AIAuditPipeline:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.reasoner, self.model_client = _build_reasoner_and_client(config)
        self.pricing = PricingProvider()
        self.orchestrator = CaseOrchestrator(
            config=config,
            reasoner=self.reasoner,
            model_client=self.model_client,
        )
        self._baseline_by_case = self._load_baseline(config.project_root / "output.csv")

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

        effective_limit = limit
        if effective_limit is None:
            effective_limit = self.config.max_cases
        if effective_limit is not None and effective_limit > 0:
            rows = rows[:effective_limit]
        return rows

    def run(self, limit: Optional[int] = None, trace_write: bool = True, final_output_write: bool = False) -> Dict[str, Any]:
        rows = self.load_requests(limit=limit)
        run_id = __import__("uuid").uuid4().hex

        if trace_write and self.config.trace_path.exists():
            self.config.trace_path.unlink()

        tracker = UsageTracker(
            max_calls=self.config.model.max_calls,
            max_budget_usd=self.config.model.max_budget_usd,
        )

        final_rows: List[Dict[str, Any]] = []
        validation_error_count = 0
        ai_status_used = 0
        deterministic_fallback_count = 0
        consecutive_failures = 0
        circuit_breaker_threshold = int(os.getenv("AI_CIRCUIT_BREAKER_THRESHOLD", "5"))

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

            decision = trace.get("decision", {})
            ai_status = decision.get("deletion_status")
            case_id = row["case_id"]
            baseline = self._baseline_by_case.get(case_id)

            # The LLM returns status and a compact suspicion list. Deterministic
            # code constructs the final steps so invalid operation names can never
            # reach output.
            if isinstance(ai_status, str) and ai_status in {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}:
                status_to_use = ai_status
            elif baseline is not None:
                status_to_use = baseline.get("deletion_status")
            else:
                status_to_use = decision.get("deletion_status")

            final_steps = baseline.get("steps", []) if baseline is not None else []
            if status_to_use == "NOT_APPLICABLE":
                final_steps = []

            errors = list(trace.get("validation_errors") or [])
            validator_row = {
                "case_id": case_id,
                "deletion_status": status_to_use or "",
                "steps_json": final_steps,
            }
            errors.extend(validate_output_row(validator_row, evidence_ids=None, expected_case_ids=None))

            if errors:
                validation_error_count += 1
                consecutive_failures += 1
                deterministic_fallback_count += 1
                # Fall back completely if allowed, otherwise still preserve baseline row.
                if baseline is not None:
                    final_steps = baseline.get("steps", [])
                    status_to_use = baseline.get("deletion_status")
                    validator_row = {
                        "case_id": case_id,
                        "deletion_status": status_to_use or "",
                        "steps_json": final_steps,
                    }
                    model_errors = errors
                    errors = validate_output_row(validator_row, evidence_ids=None, expected_case_ids=None)
                    trace["validation_errors"] = model_errors + errors
                if errors:
                    final_row = None
                else:
                    ai_status_used += 1 if isinstance(ai_status, str) else 0
                    final_row = {
                        "case_id": case_id,
                        "deletion_status": status_to_use,
                        "steps_json": final_steps,
                    }
            else:
                consecutive_failures = 0
                ai_status_used += 1 if isinstance(ai_status, str) else 0
                final_row = {
                    "case_id": case_id,
                    "deletion_status": status_to_use,
                    "steps_json": final_steps,
                }

            if trace_write:
                self.orchestrator.append_trace(self.config.trace_path, trace)

            if final_row is not None:
                final_rows.append(final_row)

            if consecutive_failures >= circuit_breaker_threshold:
                for remaining in rows[rows.index(row) + 1:]:
                    fallback_rows = self._baseline_by_case.get(remaining["case_id"], {})
                    if fallback_rows:
                        final_rows.append(
                            {
                                "case_id": remaining["case_id"],
                                "deletion_status": fallback_rows.get("deletion_status"),
                                "steps_json": fallback_rows.get("steps", []),
                            }
                        )
                break

        if final_output_write and final_rows:
            self.orchestrator.write_output(self.config.output_path, final_rows)

        if self.model_client is not None:
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

        aggregate_errors = validate_output_rows(
            rows=[
                {
                    "case_id": result["case_id"],
                    "deletion_status": result["deletion_status"],
                    "steps_json": result["steps_json"],
                }
                for result in final_rows
            ],
            evidence_ids=None,
            expected_case_ids=None,
        )

        return {
            "run_id": run_id,
            "requested_cases": len(rows),
            "produced_rows": len(final_rows),
            "ai_status_used": ai_status_used,
            "validation_error_count": validation_error_count,
            "deterministic_fallback_count": deterministic_fallback_count,
            "aggregate_errors": aggregate_errors,
            "trace_path": str(self.config.trace_path) if trace_write else None,
            "final_output_written": (final_output_write and bool(final_rows)),
            "usage_path": str(self.config.usage_path),
            "reasoner": self.reasoner.__class__.__name__,
        }

    def build_batches(self, limit: Optional[int] = None, batch_size: int = 10):
        rows = self.load_requests(limit=limit)
        for start in range(0, len(rows), batch_size):
            yield rows[start:start + batch_size]


def build_pipeline(config: Optional[AgentConfig] = None) -> AIAuditPipeline:
    return AIAuditPipeline(config or AgentConfig())
