"""Client for a LOCAL OpenAI-compatible inference server (e.g. Magnitude, llama.cpp server).

Default endpoint is Magnitude's: http://127.0.0.1:10100/inference/v1 (no API key needed locally).
Standard library only (urllib): no new dependency. By design the client refuses non-local hosts,
so study data never leave the machine.

The model only ever receives the agent-visible prompt (question, study metadata, data profile,
locked protocol). It never sees answer keys and never computes statistics.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://127.0.0.1:10100/inference/v1"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


class LLMError(RuntimeError):
    pass


@dataclass
class CallStats:
    calls: int = 0
    failures: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latencies: list[float] = field(default_factory=list)


class OpenAICompatClient:
    """`generate(prompt) -> text` for nanostat.planner.LLMPlanner."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 384,
        timeout: float = 300.0,
        seed: int | None = None,
        system: str | None = None,
    ) -> None:
        host = urlparse(base_url).hostname or ""
        if host not in LOCAL_HOSTS:
            raise LLMError(f"refusing non-local LLM endpoint {base_url!r}: data must stay on this machine")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.models_url = base_url.rstrip("/") + "/models"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.seed = seed
        self.system = system
        self.stats = CallStats()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - local host enforced above
            return json.loads(r.read().decode())

    def list_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(self.models_url, timeout=10) as r:  # noqa: S310
                data = json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise LLMError(f"cannot reach {self.models_url}: {e}") from e
        return [m.get("id", "") for m in data.get("data", [])]

    def generate(self, prompt: str) -> str:
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        t0 = time.perf_counter()
        self.stats.calls += 1
        try:
            data = self._post(payload)
            text = data["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
            self.stats.failures += 1
            # an unreachable or malformed server is an operational failure, not a crash
            return f"[LLM_ERROR] {type(e).__name__}: {e}"
        finally:
            dt = time.perf_counter() - t0
            self.stats.seconds += dt
            self.stats.latencies.append(dt)
        usage = data.get("usage") or {}
        self.stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.stats.completion_tokens += int(usage.get("completion_tokens") or 0)
        return text
