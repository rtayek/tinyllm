from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, cast

from llm.serialization_types import ConfigPayload


@dataclass(frozen=True)
class EvalResult:
    name: str
    split: str
    loss: float
    nTokens: int | None = None
    nWindows: int | None = None
    method: str = "sampled"
    checkpoint: str | None = None
    corpus: str | None = None
    notes: str | None = None
    perplexity: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "perplexity", math.exp(self.loss))

    def toDict(self) -> ConfigPayload:
        return asdict(self)

    @classmethod
    def fromDict(cls, data: ConfigPayload) -> "EvalResult":
        return cls(
            name=str(data["name"]),
            split=str(data["split"]),
            loss=float(data["loss"]),
            nTokens=_optional_int(data.get("nTokens")),
            nWindows=_optional_int(data.get("nWindows")),
            method=str(data.get("method", "sampled")),
            checkpoint=_optional_str(data.get("checkpoint")),
            corpus=_optional_str(data.get("corpus")),
            notes=_optional_str(data.get("notes")),
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return cast(str, value)
