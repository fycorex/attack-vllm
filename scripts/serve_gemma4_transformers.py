#!/usr/bin/env python3
"""Minimal OpenAI-compatible Gemma 4 image-VQA fallback server.

vLLM 0.10.2 is the CUDA-12.8-compatible serving release in this pilot, but
its supported Transformers range predates the ``gemma4`` architecture.  This
server keeps the replay protocol unchanged while recording the implementation
fallback in its startup metadata.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def content_parts(payload: dict) -> tuple[Image.Image, str]:
    content = payload["messages"][-1]["content"]
    image_data = next(item["image_url"]["url"] for item in content if item.get("type") == "image_url")
    question = next(item["text"] for item in content if item.get("type") == "text")
    encoded = image_data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"), question


class GemmaServer(ThreadingHTTPServer):
    processor: AutoProcessor
    model: AutoModelForImageTextToText
    served_model: str


class Handler(BaseHTTPRequestHandler):
    server: GemmaServer

    def _json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers(); self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.server.served_model, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"}); return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            image, prompt = content_parts(payload)
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            inputs = self.server.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to("cuda")
            started = time.monotonic()
            with torch.inference_mode():
                generated = self.server.model.generate(**inputs, max_new_tokens=int(payload.get("max_tokens", 16)), do_sample=False)
            answer = self.server.processor.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            self._json(200, {"id": f"gemma4-{time.time_ns()}", "object": "chat.completion", "model": self.server.served_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": int(inputs["input_ids"].numel()), "completion_tokens": int(generated.shape[-1] - inputs["input_ids"].shape[-1])},
                "system_fingerprint": "transformers-fallback", "latency_seconds": time.monotonic() - started})
        except Exception as error:  # return a protocol error without killing the server
            self._json(500, {"error": {"message": str(error), "type": type(error).__name__}})

    def log_message(self, format: str, *args: object) -> None:
        print(format % args, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", default="T1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16)
    model.eval().requires_grad_(False).to("cuda")
    server = GemmaServer(("127.0.0.1", args.port), Handler)
    server.processor, server.model, server.served_model = processor, model, args.served_model_name
    print(json.dumps({"implementation": "transformers-fallback", "model": str(args.model), "served_model": args.served_model_name}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
