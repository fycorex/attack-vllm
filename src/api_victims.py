from __future__ import annotations

from base64 import b64encode
from io import BytesIO
import json
import os
import time
from typing import Any, Callable
from urllib import error, request

from PIL import Image

from multimodal_victim import VictimResponse


def _png_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, "PNG")
    return b64encode(buf.getvalue()).decode("ascii")


class HTTPVictim:
    provider = ""
    default_api_key_env = ""
    default_base_url = ""

    def __init__(self, config: dict[str, Any], transport: Callable | None = None):
        self.config = dict(config)
        self.model_id = str(config["model_id"])
        self.api_key_env = str(config.get("api_key_env", self.default_api_key_env))
        self.base_url = str(config.get("base_url", self.default_base_url)).rstrip("/")
        self.timeout = float(config.get("timeout_seconds", 90))
        self._transport = transport or self._urlopen

    def _api_key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise RuntimeError(f"{self.api_key_env} is not set for {self.provider} API evaluation")
        return value

    @staticmethod
    def _urlopen(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        req = request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read())
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc

    def _request(self, image: Image.Image, prompt: str) -> tuple[str, dict, dict]:
        raise NotImplementedError

    def generate(self, image: Image.Image, prompt: str) -> VictimResponse:
        started = time.monotonic()
        text, parsed, raw = self._request(image, prompt)
        return VictimResponse(
            provider=self.provider,
            requested_model_id=self.model_id,
            resolved_model_id=parsed.get("model"),
            text=text.strip(),
            refusal=bool(parsed.get("refusal")),
            finish_reason=parsed.get("finish_reason"),
            latency_seconds=time.monotonic() - started,
            usage=parsed.get("usage", {}),
            metadata={k: v for k, v in raw.items() if k not in {"choices", "content", "candidates"}},
        )

    def close(self) -> None:
        return None


class OpenAIVictim(HTTPVictim):
    provider = "openai"
    default_api_key_env = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com/v1"

    def _request(self, image, prompt):
        payload: dict[str, Any] = {"model": self.model_id, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png_b64(image)}"}},
        ]}]}
        if self.config.get("max_output_tokens") is not None:
            payload["max_completion_tokens"] = int(self.config["max_output_tokens"])
        for key in ("temperature", "top_p"):
            if self.config.get(key) is not None:
                payload[key] = self.config[key]
        if self.config.get("reasoning_effort") is not None:
            payload["reasoning_effort"] = self.config["reasoning_effort"]
        raw = self._transport(f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}, payload, self.timeout)
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return str(msg.get("content") or ""), {"model": raw.get("model"), "refusal": bool(msg.get("refusal")), "finish_reason": choice.get("finish_reason"), "usage": raw.get("usage") or {}}, raw


class GeminiVictim(HTTPVictim):
    provider = "gemini"
    default_api_key_env = "GEMINI_API_KEY"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _request(self, image, prompt):
        generation: dict[str, Any] = {}
        mapping = {"max_output_tokens": "maxOutputTokens", "temperature": "temperature", "top_p": "topP", "top_k": "topK"}
        for source, target in mapping.items():
            if self.config.get(source) is not None:
                generation[target] = self.config[source]
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": _png_b64(image)}}]}]}
        if generation:
            payload["generationConfig"] = generation
        raw = self._transport(f"{self.base_url}/models/{self.model_id}:generateContent?key={self._api_key()}", {"Content-Type": "application/json"}, payload, self.timeout)
        candidate = (raw.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in (candidate.get("content") or {}).get("parts", []))
        usage = raw.get("usageMetadata") or {}
        return text, {"model": raw.get("modelVersion"), "refusal": not bool(text) and bool(candidate.get("finishReason")), "finish_reason": candidate.get("finishReason"), "usage": usage}, raw


class AnthropicVictim(HTTPVictim):
    provider = "anthropic"
    default_api_key_env = "ANTHROPIC_API_KEY"
    default_base_url = "https://api.anthropic.com/v1"

    def _request(self, image, prompt):
        payload: dict[str, Any] = {"model": self.model_id, "max_tokens": int(self.config.get("max_output_tokens", 64)), "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(image)}},
            {"type": "text", "text": prompt},
        ]}]}
        # Sampling is opt-in; models that reject it can omit it in YAML.
        for key in ("temperature", "top_p", "top_k"):
            if self.config.get(key) is not None:
                payload[key] = self.config[key]
        raw = self._transport(f"{self.base_url}/messages", {"x-api-key": self._api_key(), "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, payload, self.timeout)
        blocks = raw.get("content") or []
        text = "".join(str(block.get("text", "")) for block in blocks if block.get("type") == "text")
        refusal = any(block.get("type") == "refusal" for block in blocks)
        return text, {"model": raw.get("model"), "refusal": refusal, "finish_reason": raw.get("stop_reason"), "usage": raw.get("usage") or {}}, raw


PROVIDERS = {"openai": OpenAIVictim, "gemini": GeminiVictim, "anthropic": AnthropicVictim}


def create_api_victim(config: dict[str, Any], transport=None) -> HTTPVictim:
    provider = str(config.get("provider", "")).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider!r}")
    return PROVIDERS[provider](config, transport=transport)
