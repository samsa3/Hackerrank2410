"""Model-client layer for the governed AI audit agent."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class ModelCall:
    provider: str
    model: str
    content: str
    parsed_json: Optional[Dict[str, Any]]
    input_tokens_estimate: int
    output_tokens_estimate: int
    latency_ms: float
    error: Optional[str] = None


class ModelClient(Protocol):
    def complete(self, case_id, system_prompt, user_prompt, response_schema=None, options=None) -> ModelCall:
        ...


@dataclass
class UsageEntry:
    case_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    succeeded: bool


class PricingProvider:
    input_per_1m_tokens: float = 0.0
    output_per_1m_tokens: float = 0.0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000.0 * self.input_per_1m_tokens +
                output_tokens / 1_000_000.0 * self.output_per_1m_tokens)


class UsageTracker:
    def __init__(self, max_calls: int = 1000, max_budget_usd: float = 0.0) -> None:
        self.max_calls = max_calls
        self.max_budget_usd = max_budget_usd
        self.entries: list[UsageEntry] = []

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
    def __init__(self, base_url=None, model="qwen3.5:9b-q4_K_M", timeout_seconds=120):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.think = os.getenv("OLLAMA_THINK", "false").strip().lower() in {"1", "true", "yes"}
        try:
            max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
        except (TypeError, ValueError):
            max_tokens = 1024
        self.max_output_tokens = max(256, min(max_tokens, 4096))

    def health_check(self, attempts=3, delay_seconds=1.0):
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

    @staticmethod
    def extract_json(text: str):
        if not text:
            return None
        # Try direct parse first; many Ollama responses already are clean JSON.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Remove common wrappers, markdown fences, and thinking sections.
        cleaned = text.replace("```json", "").replace("```", "")
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"`{3,}", "", cleaned)

        # Locate the outermost JSON object or array.
        start_obj, obj = cleaned.find("{"), None
        start_arr, arr = cleaned.find("["), None
        if start_obj >= 0:
            end = cleaned.rfind("}")
            if end > start_obj:
                try:
                    return json.loads(cleaned[start_obj:end + 1])
                except json.JSONDecodeError:
                    pass
        if start_arr >= 0:
            end = cleaned.rfind("]")
            if end > start_arr:
                try:
                    return json.loads(cleaned[start_arr:end + 1])
                except json.JSONDecodeError:
                    pass
        return None

    def complete(self, case_id=None, system_prompt="", user_prompt="", response_schema=None, options=None):
        started = time.time()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": 0.0,
                "num_predict": min(self.max_output_tokens, 4096),
                **(options or {}),
            },
        }
        if response_schema is not None:
            payload["format"] = response_schema

        try:
            request = urllib.request.Request(
                self.base_url + "/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body.get("message", {}).get("content", "")
            elapsed = (time.time() - started) * 1000.0
            parsed = self.extract_json(content)
            return ModelCall(
                provider="ollama",
                model=self.model,
                content=content,
                parsed_json=parsed,
                input_tokens_estimate=max(1, (len(system_prompt) + len(user_prompt)) // 4),
                output_tokens_estimate=max(1, len(content) // 4),
                latency_ms=elapsed,
                error=None,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return ModelCall(provider="ollama", model=self.model, content="", parsed_json=None,
                             input_tokens_estimate=0, output_tokens_estimate=0,
                             latency_ms=(time.time() - started) * 1000.0,
                             error=f"HTTP {exc.code}: {detail[:500]}")
        except Exception as exc:
            return ModelCall(provider="ollama", model=self.model, content="", parsed_json=None,
                             input_tokens_estimate=0, output_tokens_estimate=0,
                             latency_ms=(time.time() - started) * 1000.0, error=str(exc))


def create_model_client_from_env():
    provider = os.getenv("MODEL_PROVIDER", "ollama")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    pricing = PricingProvider()
    return OllamaModelClient(base_url=base_url, model=os.getenv("OLLAMA_MODEL", "qwen3.5:9b-q4_K_M"), timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))), pricing
