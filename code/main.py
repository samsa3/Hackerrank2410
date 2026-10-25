"""Terminal entry point for the challenge solution."""

from __future__ import annotations

import os
import runpy
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "code" / "ai_agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "ai_agent"))


def _limit_from_env() -> int | None:
    value = os.getenv("MAIN_MAX_CASES") or os.getenv("MAX_CASES")
    if not value:
        return None
    try:
        number = int(value)
    except ValueError:
        return None
    return number if number > 0 else None


def _run_sample_check(config) -> int:
    import csv
    import json

    sample_path = config.project_root / "dataset" / "sample_deletion_requests.csv"
    expected_path = config.project_root / "dataset" / "sample_output.csv"
    expected_by_case = {}
    with expected_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            expected_by_case[row["case_id"]] = row

    from code.ai_agent.pipeline import build_pipeline

    pipeline = build_pipeline(config)
    from code.ai_agent.model_clients import UsageTracker

    usage_tracker = UsageTracker()
    run_id = __import__("uuid").uuid4().hex
    with sample_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    mismatches = []
    for row in rows:
        _trace, result = pipeline.orchestrator.run_case(row, run_id=run_id, usage_tracker=usage_tracker)
        expected = expected_by_case.get(row["case_id"])
        if not expected:
            continue
        try:
            actual_steps = json.loads(result.get("steps_json", "[]")) if isinstance(result, dict) else []
        except (json.JSONDecodeError, TypeError):
            actual_steps = None
        try:
            expected_steps = json.loads(expected.get("steps_json", "[]"))
        except json.JSONDecodeError:
            expected_steps = None
        if result.get("deletion_status") != expected.get("deletion_status"):
            mismatches.append(
                f"{row['case_id']}: status {result.get('deletion_status')} vs {expected.get('deletion_status')}"
            )
        if actual_steps != expected_steps:
            mismatches.append(f"{row['case_id']}: step_json differs from sample")

    print(f"[sample-check] rows={len(rows)} expected_cases={len(expected_by_case)} mismatches={len(mismatches)}")
    for item in mismatches[:20]:
        print("[sample-check mismatch]", item)
    return 0 if not mismatches else 1


def main() -> int:
    try:
        from code.ai_agent.config import default_config
        from code.ai_agent.model_clients import (
            create_model_client_from_env,
            OllamaModelClient,
            UsageTracker,
        )
        from code.ai_agent.pipeline import build_pipeline
        from code.ai_agent.proofs import promote_output_atomically

        config = default_config()
        if os.getenv("MAIN_SAMPLE_CHECK", "0").lower() in {"1", "true", "yes", "on"}:
            return _run_sample_check(config)
        client, _pricing = create_model_client_from_env()
        component_id = config.model.resolved_provider()
        base_url = config.model.resolved_base_url()
        model_name = config.model.resolved_model()

        print(f"[AI] Provider: {component_id}")
        print(f"[AI] Model: {model_name}")
        print(f"[AI] Endpoint: {base_url}")
        if isinstance(client, OllamaModelClient):
            health_error = client.health_check()
            if health_error is not None:
                print(f"[AI] Ollama endpoint is unreachable: {health_error}")
                print("[AI] falling back to deterministic pipeline.")
                raise RuntimeError(f"Ollama endpoint unreachable: {health_error}")
        print("[AI] Endpoint health check passed.")

        pipeline = build_pipeline(config)
        all_rows = pipeline.load_requests(limit=None)
        expected_rows = min(_limit_from_env() or len(all_rows), len(all_rows))

        result = pipeline.run(
            limit=_limit_from_env(),
            trace_write=True,
            final_output_write=True,
        )

        produced_rows = result.get("produced_rows", 0)
        aggregate_errors = result.get("aggregate_errors", [])

        print(
            "[AI] requested={requested}, produced={produced}, status_success={ok}, cached_rows={missing}".format(
                requested=result.get("requested_cases"),
                produced=produced_rows,
                ok=result.get("produced_rows", 0) - result.get("validation_error_count", 0),
                missing=result.get("requested_cases", 0) - produced_rows,
            )
        )

        if aggregate_errors:
            print("[AI] Validation errors:", aggregate_errors[:3])

        if produced_rows == expected_rows and produced_rows > 0 and aggregate_errors == []:
            ai_output = config.output_path
            if ai_output.exists() and ai_output.is_file():
                if os.getenv("MAIN_PROMOTE_ROOT_OUTPUT", "0").lower() in {"1", "true", "yes"}:
                    manifest = promote_output_atomically(
                        source=ai_output,
                        destination=ROOT / "output.csv",
                        valid_rows=produced_rows,
                    )
                    print(
                        "[AI] Promoted output.csv atomically: "
                        f"rows={manifest.valid_rows}, digest={manifest.digest}"
                    )
                print(f"[AI] Wrote {produced_rows} valid rows to {config.output_path}")
                return 0
    except Exception:
        print("[AI] falling back to deterministic baseline after AI error.")
        traceback.print_exc()
        Path(ROOT / "ai_agent_failure.txt").write_text(
            __import__("traceback").format_exc(),
            encoding="utf-8",
        )

    runpy.run_path(str(ROOT / "code" / "deterministic_main.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
