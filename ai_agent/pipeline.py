"""Pipeline orchestration for the AI deletion-audit agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_agent.config import AgentConfig
from ai_agent.model_clients import create_model_client_from_env, UsageTracker, PricingProvider
from ai_agent.orchestration import CaseOrchestrator
from ai_agent.reasoner import DeterministicFallbackReasoner, SchemaConstrainedModelReasoner
from ai_agent.validators import validate_output_rows

_TRACE_NAME = "ai_agent_run_trace.jsonl"


def _build_reasoner_and_client(config: AgentConfig):
    provider = config.model.resolved_provider()
    if provider in {"placeholder", "deterministic-baseline", "none"}:
        return DeterministicFallbackReasoner(), None

    client, pricing = create_model_client_from_env()
    from ai_agent.model_clients import OllamaModelClient

    if isinstance(client, OllamaModelClient):
        health_error = client.health_check()
        if health_error is not None:
            return DeterministicFallbackReasoner(), None

    return SchemaConstrainedModelReasoner(max_repair_calls=config.model.max_retries), client


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
        # If the pipeline defaults fail closed, replace prior run artifacts.
        if trace_write:
            trace_path = self.config.trace_path
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            if trace_path.exists():
                trace_path.unlink()

        tracker = UsageTracker(
            max_calls=self.config.model.max_calls,
            max_budget_usd=self.config.model.max_budget_usd,
        )
        run_id: str = __import__("uuid").uuid4().hex

        successful_rows: List[Dict[str, str]] = []
        validation_error_count = 0

        for row in rows:
            trace, result_row = self.orchestrator.run_case(row, run_id=run_id, usage_tracker=tracker)
            if trace_write:
                self.orchestrator.append_trace(self.config.trace_path, trace)
            if trace.get("validation_errors"):
                validation_error_count += 1
            if result_row is not None:
                successful_rows.append(result_row)

        if final_output_write and successful_rows:
            self.orchestrator.write_output(self.config.output_path, successful_rows)

        # Persist usage immediately; it is too easy to lose this afterward.
        usage_summary = tracker.summary()
        with self.config.usage_path.open("w", encoding="utf-8") as handle:
            import json

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
                    "summary": usage_summary,
                    "estimated_cost_usd": self.pricing.estimate_cost(
                        usage_summary["input_tokens_estimate"],
                        usage_summary["output_tokens_estimate"],
                    ),
                },
                handle,
                indent=2,
            )

        validator_rows = [
            {
                "case_id": result["case_id"],
                "deletion_status": result["deletion_status"],
                "steps_json": result["steps_json"],
            }
            for result in successful_rows
        ]

        aggregate_errors = validate_output_rows(
            rows=validator_rows,
            evidence_ids=None,
            expected_case_ids=None,
        )

        return {
            "run_id": run_id,
            "requested_cases": len(rows),
            "produced_rows": len(successful_rows),
            "validation_error_count": validation_error_count,
            "aggregate_errors": aggregate_errors,
            "trace_path": str(self.config.trace_path) if trace_write else None,
            "final_output_written": final_output_write and bool(successful_rows),
            "usage_path": str(self.config.usage_path),
            "reasoner": self.reasoner.__class__.__name__,
        }

    def build_batches(self, limit: Optional[int] = None, batch_size: int = 10):
        rows = self.load_requests(limit=limit)
        for start in range(0, len(rows), batch_size):
            yield rows[start:start + batch_size]


def build_pipeline(config: Optional[AgentConfig] = None) -> AIAuditPipeline:
    return AIAuditPipeline(config or AgentConfig())
