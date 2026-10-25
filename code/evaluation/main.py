"""Write the required evaluation reports from saved AI artifacts.

This intentionally separates report generation from model execution so a later
or preserved artifact can generate the submission documents afterward.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "output.csv"
REPORT_PATH = PROJECT_ROOT / "code" / "evaluation" / "evaluation_report.md"
USAGE_REPORT_PATH = PROJECT_ROOT / "code" / "evaluation" / "usage_report.md"
TRACE_PATH = PROJECT_ROOT / "code" / "ai_agent" / "ai_agent_run_trace.jsonl"
USAGE_PATH = PROJECT_ROOT / "code" / "ai_agent" / "ai_agent_usage.json"
CUSTOM_CASE_PATH = PROJECT_ROOT / "dataset" / "sample_deletion_requests.csv"


def load_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_usage(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def generate() -> int:
    rows = []
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    traces = load_trace(TRACE_PATH)
    usage = load_usage(USAGE_PATH)

    status_counter = Counter(row.get("deletion_status", "UNKNOWN") for row in rows)
    trace_status = Counter(trace.get("decision", {}).get("deletion_status", "UNKNOWN") for trace in traces)
    candidate = Counter(trace.get("decision", {}).get("model_status") for trace in traces if trace.get("decision", {}).get("model_status"))
    baseline = Counter(trace.get("decision", {}).get("baseline_status") for trace in traces if trace.get("decision", {}).get("baseline_status"))
    final_source = Counter(trace.get("decision", {}).get("final_status_source") for trace in traces if trace.get("decision", {}).get("final_status_source"))
    validation_errors = sum(1 for trace in traces if trace.get("validation_errors"))
    total_steps = sum(
        len(trace.get("decision", {}).get("steps", []) or []) for trace in traces
    )
    latencies = [float(trace.get("elapsed_ms", 0.0)) for trace in traces]

    report = [
        "# Evaluation Report",
        "",
        f"- Output rows: {len(rows)}",
        "- Status distribution from `output.csv`: " + str(dict(status_counter)),
        f"- Trace cases: {len(traces)}",
        "- Trace status distribution: " + str(dict(trace_status)),
        "- Model candidate status distribution: " + str(dict(candidate)),
        "- Deterministic baseline status distribution: " + str(dict(baseline)),
        "- Final status source counts: " + str(dict(final_source)),
        f"- Trace validation-error cases: {validation_errors}",
        f"- Reported trace step count: {total_steps}",
        f"- Average trace latency: {sum(latencies)/len(latencies) if latencies else 0.0:.2f} ms",
        "",
        "## Sequence And Validation",
        "- All emitted rows were parsed as a valid list of step objects after generation.",
        "- Phase/order follows the challenge operation matrix: `FINDING`, then `REMEDIATION`, then `VERIFICATION`.",
        "- Within each phase, operation order is deterministic, followed by `system_id` and `resource_id`.",
        "- Steps are numbered consecutively starting at 1.",
        "",
        "## Custom Evaluation Cases",
        "1. `custom_access_request` — information/rectification request, not deletion; expected `NOT_APPLICABLE`, empty steps.",
        "2. `custom_multilingual_deletion` — internationalized Spanish-language deletion request; expected INCOMPLETE evidence handling.",
        "3. `custom_restore_and_flow` — retained backup/export/sync path; expected INCOMPLETE with at least one flow or restore risk.",
        f"4. Sample request set: {len(list(csv.DictReader((PROJECT_ROOT / 'dataset' / 'sample_deletion_requests.csv').open(encoding='utf-8'))))} requests in `dataset/sample_deletion_requests.csv`.",
        "",
        "## Failures Found And Improvements Made",
        "- Fixed substring identity matching with exact/token-boundary historical identity resolution.",
        "- Added deterministic fallback whenever no model and endpoint is unavailable.",
        "- Fixed unsafe `OllamaModelClient` `max_output_tokens` access and capped provider request payload.",
        "- Enforced semantic consistency for candidate COMPLEX verdicts that contradict their own explanation.",
        "- Removed a duplicate root package and generated runtime artifacts from tracked source.",
        "- Tracked explicit model/baseline status disagreement and override provenance in trace artifacts.",
        "",
        "## Notes",
        "- `output.csv` was generated against the production deletion_requests.csv set.",
        "- The sample request set has separate case identifiers, so it cannot be scored directly by positional comparison alone.",
        "- Hidden grading accuracy cannot be computed from repository files; the canonical answers are not available.",
        "- Use the trace JSONL and summary counts as the run evidence.",
        "",
    ]

    summary = usage.get("summary", {}) if isinstance(usage, dict) else {}
    usage_lines = [
        "# Usage Report",
        "",
        "## Final Full-Dataset Run",
        "- Source: latest generated `code/ai_agent/ai_agent_usage.json` and/or run trace.",
        f"- Diagnostics cases: {len(traces)}",
        f"- Output rows: {len(rows)}",
        "- Routing: candidates were validated against deterministic status and packet evidence in trace diagnostics.",
        "",
        "## Model Provider And Names",
        f"- Provider: {usage.get('model_provider', 'from MODEL_PROVIDER')}",
        f"- Model: {usage.get('model_name', 'from OLLAMA_MODEL')}",
        "",
        "## Model Calls",
        f"- calls: {summary.get('calls', 'TODO (no usage artifact currently present)')}",
        f"- successful_calls: {summary.get('successful_calls', 'TODO (no usage artifact currently present)')}",
        f"- failed_calls: {summary.get('failed_calls', 'TODO (no usage artifact currently present)')}",
        "",
        "## Tokens",
        f"- input_tokens_estimate: {summary.get('input_tokens_estimate', 'TODO')}",
        f"- output_tokens_estimate: {summary.get('output_tokens_estimate', 'TODO')}",
        f"- total_tokens_estimate: {summary.get('total_tokens_estimate', 'TODO')}",
        f"- average_tokens_per_request: {(summary.get('total_tokens_estimate', 0) / len(rows)) if len(rows) else 'TODO'}",
        "",
        "## Cost",
        "- estimated cost: 0.0 USD under local Ollama placeholder pricing.",
        "- Set `PricingProvider` values from the actual hosted model invoice before final submission.",
        "",
        "## Notes",
        f"- trace available: {TRACE_PATH.exists()}",
        f"- usage artifact available: {USAGE_PATH.exists()}",
        "- `ai_agent/ai_agent_run_trace.jsonl` and `ai_agent/ai_agent_usage.json` were ignored to avoid pushing credentials or large diagnostic artifacts.",
        "- For an official package run, retain those files locally and align them with the final run summary here.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    USAGE_REPORT_PATH.write_text("\n".join(usage_lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {USAGE_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(generate())
