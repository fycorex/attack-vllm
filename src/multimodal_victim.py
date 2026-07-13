from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from PIL import Image


@dataclass
class VictimResponse:
    provider: str
    requested_model_id: str
    text: str = ""
    resolved_model_id: str | None = None
    refusal: bool = False
    finish_reason: str | None = None
    latency_seconds: float = 0.0
    usage: dict[str, int | float | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultimodalVictim(Protocol):
    provider: str
    model_id: str

    def generate(self, image: Image.Image, prompt: str) -> VictimResponse: ...

    def close(self) -> None: ...
