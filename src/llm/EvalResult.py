from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from llm.serialization_types import ConfigPayload, optional_int, optional_str


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
            nTokens=optional_int(data.get("nTokens")),
            nWindows=optional_int(data.get("nWindows")),
            method=str(data.get("method", "sampled")),
            checkpoint=optional_str(data.get("checkpoint")),
            corpus=optional_str(data.get("corpus")),
            notes=optional_str(data.get("notes")),
        )
