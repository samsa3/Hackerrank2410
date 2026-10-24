"""Terminal entry point for the challenge solution.

This entry point prefers the governed AI pipeline. If no model endpoint is
available or the model returns no valid rows, it falls back to the deterministic
baseline so the submission still receives a schema-valid prediction.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ai_agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "ai_agent"))


def main() -> int:
    # Attempt the bounded AI audit pipeline first. It writes diagnostics and,
    # when model output is valid and write permission is enabled, writes
    # ai_agent_output.csv without harming the deterministic baseline.
    try:
        from ai_agent.config import default_config
        from ai_agent.pipeline import build_pipeline

        pipeline = build_pipeline(default_config())
        result = pipeline.run(limit=None, trace_write=True, final_output_write=False)
        if result.get("produced_rows", 0) > 0:
            return 0
    except Exception:
        # Any failure deliberately falls through to the deterministic baseline.
        pass

    # The baseline remains the fallback for sandbox runs without a reachable model.
    runpy.run_path(str(ROOT / "code" / "deterministic_main.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
