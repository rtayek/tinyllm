from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from llm.serialization_types import int_tuple, optional_int


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
        return cls(
            prompt=str(data["prompt"]),
            continuation=str(data["continuation"]),
            text=str(data["text"]),
            tokenIds=int_tuple(data["tokenIds"], "GeneratedCandidate tokenIds"),
            promptTokenCount=int(cast(int, data["promptTokenCount"])),
            generatedTokenCount=int(cast(int, data["generatedTokenCount"])),
            seed=optional_int(data.get("seed")),
            temperature=float(cast(float, data["temperature"])),
            topK=optional_int(data.get("topK")),
            maxNewTokens=int(cast(int, data["maxNewTokens"])),
        )
