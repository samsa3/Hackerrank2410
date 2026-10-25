from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_env_file(path: Optional[Path] = None) -> None:
    """Load simple KEY=VALUE settings into os.environ."""
    if path is None:
        path = PROJECT_ROOT / ".env"
    path = Path(path)
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


@dataclass
class ModelSettings:
    provider: str = None
    model: str = None
    base_url: str = None
    api_key_env: str = None
    temperature: float = 0.0
    max_output_tokens: int = 4096
    max_retries: int = 1
    timeout_seconds: int = 120
    max_calls: int = 1000
    max_budget_usd: float = 0.0

    def resolved_provider(self) -> str:
        return (self.provider or os.getenv("MODEL_PROVIDER") or "ollama").lower()

    def resolved_model(self) -> str:
        return (self.model or os.getenv("OLLAMA_MODEL") or "qwen3.5:9b-q4_K_M")

    def resolved_base_url(self) -> str:
        return (self.base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")


@dataclass
class RetrievalSettings:
    max_evidence_records_per_user: int = 20
    persist_packet_artifacts: bool = True
    omit_late_evidence: bool = True


@dataclass
class AgentConfig:
    project_root: Path = PROJECT_ROOT
    output_path: Path = PROJECT_ROOT / "ai_agent_output.csv"
    trace_path: Path = PROJECT_ROOT / "code" / "ai_agent" / "ai_agent_run_trace.jsonl"
    usage_path: Path = PROJECT_ROOT / "code" / "ai_agent" / "ai_agent_usage.json"
    model: ModelSettings = None
    retrieval: RetrievalSettings = None
    max_cases: int = None

    def __post_init__(self) -> None:
        load_env_file(self.project_root / ".env")
        if self.model is None:
            self.model = ModelSettings()
        if self.retrieval is None:
            self.retrieval = RetrievalSettings()

        self.max_cases = _env_int("MAX_CASES", self.max_cases)

        if self.model.max_calls == 1000 and self.max_cases is not None:
            self.model.max_calls = max(1, self.max_cases)

        budget = os.getenv("MAX_BUDGET_USD")
        if budget is not None:
            try:
                self.model.max_budget_usd = float(budget)
            except ValueError:
                pass

        max_output = os.getenv("MAX_OUTPUT_TOKENS")
        if max_output is not None:
            try:
                parsed = int(max_output)
                self.model.max_output_tokens = min(parsed, 4096)
            except ValueError:
                pass


def default_config() -> AgentConfig:
    return AgentConfig()
