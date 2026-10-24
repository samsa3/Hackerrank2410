"""Terminal entry point for the challenge solution.

This entry point prefers the governed AI pipeline.  If it succeeds for the full
case set, its generated file replaces the root output.csv.  Otherwise the entry
point falls back to the deterministic baseline so a schema-valid baseline remains.
"""

from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "code/ai_agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "code/ai_agent"))


def main() -> int:
    # A simple dry-run size lets you test the new pipeline without waiting
    # for all 518 model calls.
    limit_string = os.getenv("MAIN_MAX_CASES")
    case_limit = int(limit_string) if limit_string and limit_string.isdigit() else None
    write_output = os.getenv("MAIN_WRITE_OUTPUT", "false").lower() in {"1", "true", "yes"}

    try:
        from code.ai_agent.config import default_config
        from code.ai_agent.pipeline import build_pipeline

        config = default_config()
        pipeline = build_pipeline(config)
        all_rows = pipeline.load_requests(limit=None)
        result = pipeline.run(limit=case_limit, trace_write=True, final_output_write=True)

        produced_rows = result.get("produced_rows", 0)
        expected_rows = len(all_rows[:case_limit]) if case_limit else len(all_rows)
        aggregate_errors = result.get("aggregate_errors", [])

        if produced_rows == expected_rows and produced_rows > 0 and aggregate_errors == []:
            ai_output = config.output_path
            if ai_output.exists() and ai_output.is_file():
                shutil.copy2(ai_output, ROOT / "output.csv")
            return 0
    except Exception:
        # Deliberately fall through to the deterministic baseline.  Leaving the
        # original output untouched on failure is preferred over partial output.
        pass

    runpy.run_path(str(ROOT / "code" / "deterministic_main.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
