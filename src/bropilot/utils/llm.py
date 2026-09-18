from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

from ..utils.common import get_logger

log = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_LEGACY_SAMPLING_MODELS = (
    "claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5", "claude-opus-4-1",
    "claude-sonnet-4-0", "claude-opus-4-0", "claude-3",
)


def supports_legacy_sampling(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _LEGACY_SAMPLING_MODELS)


class LlmError(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


def encode_image(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def extract_json(text: str) -> Any:
    candidate = text.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LlmError(f"model did not return valid JSON: {text[:200]!r}")


class LlmClient:
    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        max_retries: int = 4,
        concurrency: int = 4,
        api_key: str | None = None,
        thinking: str | None = "disabled",
    ) -> None:
        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LlmError(
                "ANTHROPIC_API_KEY is not set — export it or pass `llm.api_key` in the config"
            )
        self._client = Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.concurrency = max(1, concurrency)
        self.thinking = thinking
        self.legacy = supports_legacy_sampling(model)
        self.usage = Usage()
        self._lock = threading.Lock()

    def complete(
        self,
        blocks: Sequence[dict[str, Any]] | str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prefill: str | None = None,
    ) -> str:
        content = [{"type": "text", "text": blocks}] if isinstance(blocks, str) else list(blocks)
        use_prefill = bool(prefill) and self.legacy
        if prefill and not use_prefill:
            content.append({
                "type": "text",
                "text": f"Ответ начни сразу с символа {prefill} — без пояснений и без markdown.",
            })
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if use_prefill:
            messages.append({"role": "assistant", "content": prefill})

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": system or "",
            "messages": messages,
        }
        if self.legacy:
            params["temperature"] = self.temperature if temperature is None else temperature
        if self.thinking:
            params["thinking"] = {"type": self.thinking}

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.messages.create(**params)
                with self._lock:
                    self.usage.add(
                        Usage(
                            input_tokens=response.usage.input_tokens,
                            output_tokens=response.usage.output_tokens,
                            calls=1,
                        )
                    )
                text = "".join(b.text for b in response.content if b.type == "text")
                if getattr(response, "stop_reason", None) == "max_tokens":
                    log.warning("reply truncated at max_tokens=%s", params["max_tokens"])
                return (prefill + text) if use_prefill else text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status in (400, 401, 403, 404):
                    break
                if attempt == self.max_retries:
                    break
                delay = min(30.0, 2.0**attempt)
                log.warning("LLM call failed (%s/%s): %s — retrying in %.0fs",
                            attempt, self.max_retries, exc, delay)
                time.sleep(delay)
        raise LlmError(f"LLM call failed: {last_error}") from last_error

    def complete_json(self, blocks, **kwargs) -> Any:
        return extract_json(self.complete(blocks, prefill=kwargs.pop("prefill", None), **kwargs))

    def map(self, fn: Callable[[T], R], items: Iterable[T], desc: str = "") -> list[R]:
        items = list(items)
        if not items:
            return []
        results: list[Any] = [None] * len(items)
        errors: list[Exception] = []
        finished = [0]

        def work(pair: tuple[int, T]) -> None:
            i, item = pair
            try:
                results[i] = fn(item)
            except Exception as exc:  # noqa: BLE001
                log.error("%s: item %d failed: %s", desc or "map", i, exc)
                with self._lock:
                    errors.append(exc)
                results[i] = None
            finally:
                with self._lock:
                    finished[0] += 1
                    log.info("%s progress: %d / %d", desc or "map", finished[0], len(items))

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            list(pool.map(work, enumerate(items)))
        if errors and len(errors) == len(items):
            raise LlmError(f"{desc or 'map'}: all {len(items)} calls failed; first error: {errors[0]}")
        return results

    def report(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.usage.calls,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
        }
