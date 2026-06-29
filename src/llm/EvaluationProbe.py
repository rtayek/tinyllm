from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .EvalResult import EvalResult


@dataclass(frozen=True)
class EvalContext:
    name: str
    split: str = "validation"
    tokens: torch.Tensor | None = None
    raw: bytes | None = None
    corpus: str | None = None
    checkpoint: str | None = None
    notes: str | None = None


class EvaluationProbe(Protocol):
    def evaluate(self, context: EvalContext) -> EvalResult: ...
