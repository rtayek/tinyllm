from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .EvalResult import EvalResult


@dataclass(frozen=True)
class EvalContext:
    name: str
    # Split key as used by the underlying Evaluator / SequenceDataModule.
    # The data modules key the held-out split as "val" (not "validation"),
    # so this default must match or a probe will request a missing split.
    split: str = "val"
    tokens: torch.Tensor | None = None
    raw: bytes | None = None
    corpus: str | None = None
    checkpoint: str | None = None
    notes: str | None = None


class EvaluationProbe(Protocol):
    def evaluate(self, context: EvalContext) -> EvalResult: ...
