from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BookLossRow:
    book: str
    path: str
    seed: int
    loss: float
    perplexity: float
    method: str
    stride: int | None
    nTokens: int | None
    nWindows: int | None


@dataclass(frozen=True)
class PerBookReport:
    checkpoint: str
    device: str
    iters: int
    full_split: bool
    method: str | None
    stride: int | None
    seed: int
    books: list[BookLossRow]
    average_loss: float | None

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        if self.average_loss is None:
            data.pop("average_loss")
        return data


@dataclass(frozen=True)
class BaselineRow:
    book: str
    loss: float
    perplexity: float


@dataclass(frozen=True)
class ContextProbeRow:
    context: int
    average_loss: float
    delta: float
    delta_pct: float
    marginal: float | None


@dataclass(frozen=True)
class CorruptionBookRow:
    book: str
    baseline: float
    corrupted: float
    delta: float
    delta_pct: float


@dataclass(frozen=True)
class ExperimentRow:
    name: str
    per_book: list[CorruptionBookRow]


@dataclass(frozen=True)
class DestructionReport:
    checkpoint: str
    device: str
    iters: int
    seed: int
    block_size: int
    baseline: list[BaselineRow]
    context_probe: list[ContextProbeRow]
    experiments: list[ExperimentRow]

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NgramBookLoss:
    book: str
    loss: float


@dataclass(frozen=True)
class NgramResult:
    n: int
    average_loss: float
    per_book: list[NgramBookLoss]
    comparison: str | None


@dataclass(frozen=True)
class NgramReport:
    train: str
    train_bytes: int
    transformer_reference: dict[str, object] | None
    models: list[NgramResult]

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)
