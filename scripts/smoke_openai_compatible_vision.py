#!/usr/bin/env python3
"""Probe whether an OpenAI-compatible gateway accepts standard vision input."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib import error, request

from PIL import Image

from api_victims import OpenAIVictim
from transfer_eval import atomic_write_json


def list_models(base_url: str, key: str, timeout: float, browser_headers: bool = False) -> tuple[list[str], str | None]:
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if browser_headers:
        headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
    req = request.Request(f"{base_url.rstrip('/')}/models", headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read())
        return sorted(str(row["id"]) for row in body.get("data", []) if row.get("id")), None
    except Exception as exc:
        if isinstance(exc, error.HTTPError):
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            return [], f"HTTP {exc.code}: {detail}"
        return [], f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--image-format", choices=("png", "jpeg", "webp"), default="jpeg")
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="low")
    parser.add_argument("--browser-headers", action="store_true")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-real-api", action="store_true", help="Required: sends one real request per model.")
    args = parser.parse_args()
    if not args.allow_real_api:
        raise SystemExit("Refusing network requests without --allow-real-api")
    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"{args.api_key_env} is not set")

    advertised, list_error = list_models(args.base_url, key, args.timeout, args.browser_headers)
    image = Image.new("RGB", (32, 32), (40, 120, 220))
    rows = []
    for model in args.model:
        victim = OpenAIVictim({"model_id": model, "base_url": args.base_url, "api_key_env": args.api_key_env,
            "image_format": args.image_format, "image_detail": args.image_detail,
            "browser_headers": args.browser_headers, "max_output_tokens": 8, "timeout_seconds": args.timeout})
        try:
            response = victim.generate(image, "Reply with only the dominant color in this image.")
            rows.append({"model_id": model, "advertised": model in advertised, "vision_request_ok": True,
                "resolved_model_id": response.resolved_model_id, "finish_reason": response.finish_reason,
                "text": response.text, "error": None})
        except Exception as exc:
            rows.append({"model_id": model, "advertised": model in advertised, "vision_request_ok": False,
                "resolved_model_id": None, "finish_reason": None, "text": "",
                "error": {"type": type(exc).__name__, "message": str(exc)[:1000]}})

    report = {"created_at": datetime.now(timezone.utc).isoformat(), "base_url": args.base_url,
        "api_key_env": args.api_key_env, "api_key_serialized": False, "image_format": args.image_format,
        "image_detail": args.image_detail, "browser_headers": args.browser_headers,
        "advertised_model_count": len(advertised), "advertised_models": advertised,
        "list_models_error": list_error, "results": rows}
    atomic_write_json(Path(args.output), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
