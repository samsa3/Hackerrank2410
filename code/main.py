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


def main() -> int:
    try:
        from code.ai_agent.config import default_config
        from code.ai_agent.model_clients import (
            create_model_client_from_env,
            OllamaModelClient,
            UsageTracker,
        )
        from code.ai_agent.pipeline import build_pipeline

        config = default_config()
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
                    shutil.copy2(ai_output, ROOT / "output.csv")
                    print("[AI] Promoted to root output.csv because MAIN_PROMOTE_ROOT_OUTPUT=1")
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
