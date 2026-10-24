"""Model-client layer for the Step 2 AI-powered audit agent.

The goal of this module is to isolate provider-specific code from packet
assembly and validation. Higher-level code asks a client for a schema-shaped
response and records usage through the budget tracker.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class ModelCall:
    """One bounded call to an LLM."""

    provider: str
    model: str
    content: str
    parsed_json: Optional[Dict[str, Any]]
    input_tokens_estimate: int
    output_tokens_estimate: int
    latency_ms: float
    error: Optional[str] = None


@dataclass
class UsageEntry:
    case_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    succeeded: bool


class ModelClient(Protocol):
    """Interface for reasoning models.

    Implementations must not mutate packet data. They should return raw content
    and let the deterministic validators decide whether it is usable.
    """

    def complete(
        self,
        case_id: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> ModelCall:
        ...


class PricingProvider:
    """Estimated pricing only."""

    # TODO: Replace these placeholders with the pricing for the chosen provider.
    # Ollama local calls are effectively free, while hosted providers charge.
    input_per_1m_tokens: float = 0.0
    output_per_1m_tokens: float = 0.0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000.0 * self.input_per_1m_tokens +
                output_tokens / 1_000_000.0 * self.output_per_1m_tokens)


class UsageTracker:
    """Per-run usage and budget tracking."""

    def __init__(self, max_calls: int = 1000, max_budget_usd: float = 0.0) -> None:
        self.max_calls = max_calls
        self.max_budget_usd = max_budget_usd
        self.entries: list[UsageEntry] = []

    def can_make_call(self) -> bool:
        return len(self.entries) < self.max_calls

    def budget_remaining(self, pricing: Optional[PricingProvider] = None) -> Optional[float]:
        if pricing is None or self.max_budget_usd <= 0:
            return None
        spent = sum(pricing.estimate_cost(e.input_tokens, e.output_tokens) for e in self.entries)
        return self.max_budget_usd - spent

    def record(self, entry: UsageEntry) -> None:
        self.entries.append(entry)

    def summary(self) -> Dict[str, Any]:
        input_tokens = sum(e.input_tokens for e in self.entries)
        output_tokens = sum(e.output_tokens for e in self.entries)
        latency = sum(e.latency_ms for e in self.entries)
        successful = sum(1 for e in self.entries if e.succeeded)
        return {
            "calls": len(self.entries),
            "successful_calls": successful,
            "failed_calls": len(self.entries) - successful,
            "input_tokens_estimate": input_tokens,
            "output_tokens_estimate": output_tokens,
            "total_tokens_estimate": input_tokens + output_tokens,
            "latency_ms": latency,
        }


class OllamaModelClient:
    """Local or hosted Ollama endpoint.

    Important note: research indicates that Ollama structured outputs are local
    endpoint capabilities, while cloud support may be limited. The client asks
    for a JSON object when a schema is available, but parsing is still validated elsewhere.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "qwen2.5:7b",
        timeout_seconds: int = 120,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def health_check(self, attempts: int = 3, delay_seconds: float = 1.0) -> Optional[str]:
        last_error = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                with urllib.request.urlopen(self.base_url + "/api/tags", timeout=5) as response:
                    if response.status == 200:
                        return None
                    last_error = f"HTTP {response.status}"
            except Exception as exc:
                last_error = str(exc)
            if attempt < attempts:
                time.sleep(delay_seconds)
        return last_error

    def complete(
        self,
        case_id: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ModelCall:
        started = time.time()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            # Keep models compact and deterministic.
            "options": {
                "temperature": 0.0,
                "num_predict": 1024,
                **(options or {}),
            },
        }

        # Ollama supports a schema in `format` on compatible local endpoints.
        if response_schema is not None:
            payload["format"] = response_schema

        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body.get("message", {}).get("content", "")
            parsed = None
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
            elapsed_ms = (time.time() - started) * 1000.0
            input_estimate = max(1, (len(system_prompt) + len(user_prompt)) // 4)
            output_estimate = max(1, len(content) // 4)
            return ModelCall(
                provider="ollama",
                model=self.model,
                content=content,
                parsed_json=parsed,
                input_tokens_estimate=input_estimate,
                output_tokens_estimate=output_estimate,
                latency_ms=elapsed_ms,
                error=None,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            elapsed_ms = (time.time() - started) * 1000.0
            return ModelCall(
                provider="ollama",
                model=self.model,
                content="",
                parsed_json=None,
                input_tokens_estimate=0,
                output_tokens_estimate=0,
                latency_ms=elapsed_ms,
                error=f"HTTP {exc.code}: {detail[:500]}",
            )
        except Exception as exc:
            elapsed_ms = (time.time() - started) * 1000.0
            return ModelCall(
                provider="ollama",
                model=self.model,
                content="",
                parsed_json=None,
                input_tokens_estimate=0,
                output_tokens_estimate=0,
                latency_ms=elapsed_ms,
                error=str(exc),
            )


def create_model_client_from_env() -> tuple[ModelClient, PricingProvider]:
    """TODO: extend for HTTP/OpenAI-compatible or Anthropic providers.

    This is deliberately simple because the active model choice is still
    one of the user-facing budget decisions from Step 2.
    """
    provider = os.getenv("MODEL_PROVIDER", "ollama").lower()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # TODO: provide a real pricing table for your chosen provider.
    pricing = PricingProvider()

    if provider in {"ollama", "beam", "beam:cloud"}:
        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        return OllamaModelClient(base_url=base_url, model=model_name), pricing

    # TODO: implement hosted provider clients with native JSON-schema support.
    raise ValueError(f"Unsupported model provider: {provider}")
