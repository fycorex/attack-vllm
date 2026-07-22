"""Small resumable OpenAI-compatible client used for VQA replay."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx

from .io import atomic_json
from .vqa_normalization import normalize_answer


VQA_PROMPT = "Answer with a short answer only. Do not explain."


def replay_key(*, target: str, image: Path, question: str, revision: str) -> str:
    return "|".join((target, str(image.resolve()), question, revision))


def ask_vqa(
    endpoint: str,
    model: str,
    image: Path,
    question: str,
    *,
    cache_path: Path | None = None,
    revision: str = "unknown",
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    key = replay_key(target=model, image=image, question=question, revision=revision)
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return cached
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            {"type": "text", "text": f"{VQA_PROMPT}\n{question}"},
        ]}],
    }
    started = time.monotonic()
    last_error: str | None = None
    for attempt in range(3):
        try:
            response = httpx.post(f"{endpoint.rstrip('/')}/v1/chat/completions", json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            body = response.json()
            raw = str(body["choices"][0]["message"]["content"]).strip()
            result = {
                "key": key,
                "raw_answer": raw,
                "normalized_answer": normalize_answer(raw),
                "latency_seconds": time.monotonic() - started,
                "usage": body.get("usage", {}),
                "response": body,
            }
            if cache_path:
                atomic_json(cache_path, result)
            return result
        except (httpx.HTTPError, KeyError, ValueError) as error:
            last_error = str(error)
            time.sleep(2**attempt)
    raise RuntimeError(f"vLLM replay failed after retries: {last_error}")
