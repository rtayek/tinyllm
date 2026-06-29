from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


def _optional_int(value: object) -> int | None:
    return None if value is None else int(cast(int, value))


@dataclass(frozen=True)
class GeneratedCandidate:
    prompt: str
    continuation: str
    text: str
    tokenIds: tuple[int, ...]
    promptTokenCount: int
    generatedTokenCount: int
    seed: int | None
    temperature: float
    topK: int | None
    maxNewTokens: int

    def toDict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "continuation": self.continuation,
            "text": self.text,
            "tokenIds": list(self.tokenIds),
            "promptTokenCount": self.promptTokenCount,
            "generatedTokenCount": self.generatedTokenCount,
            "seed": self.seed,
            "temperature": self.temperature,
            "topK": self.topK,
            "maxNewTokens": self.maxNewTokens,
        }

    @classmethod
    def fromDict(cls, data: dict[str, object]) -> "GeneratedCandidate":
        raw_token_ids = data["tokenIds"]
        if not isinstance(raw_token_ids, list):
            raise ValueError("GeneratedCandidate tokenIds must be a list")
        token_ids = cast(list[object], raw_token_ids)
        return cls(
            prompt=str(data["prompt"]),
            continuation=str(data["continuation"]),
            text=str(data["text"]),
            tokenIds=tuple(int(cast(Any, token)) for token in token_ids),
            promptTokenCount=int(cast(int, data["promptTokenCount"])),
            generatedTokenCount=int(cast(int, data["generatedTokenCount"])),
            seed=_optional_int(data.get("seed")),
            temperature=float(cast(float, data["temperature"])),
            topK=_optional_int(data.get("topK")),
            maxNewTokens=int(cast(int, data["maxNewTokens"])),
        )
