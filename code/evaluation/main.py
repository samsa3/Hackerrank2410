from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "output.csv"
REPORT_PATH = PROJECT_ROOT / "code" / "evaluation" / "evaluation_report.md"


def main() -> int:
    if not OUTPUT_PATH.exists():
        print("Root output.csv is missing; run code/main.py first.")
        return 1
    print("evaluation/evaluation_report.md contains placeholder metrics and custom cases.")
    print("Usage totals should be aligned against the saved run trace:")
    print(PROJECT_ROOT / "ai_agent" / "ai_agent_run_trace.jsonl")
    print("Manual review is required before submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
