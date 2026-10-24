from __future__ import annotations

import argparse
from pathlib import Path

from ai_agent.config import default_config
from ai_agent.pipeline import build_pipeline


def main() -> int:
    config = default_config()
    pipeline = build_pipeline(config)
    results = pipeline.run()

    print(f"Prepared {len(results)} queued cases for the Step 2 AI-agent implementation.")
    print(f"Output will ultimately be written to: {config.output_path}")
    print("No production output was written yet; Step 1 only establishes architecture.")
    return 0


if __name__ == "__main__":
    main()
