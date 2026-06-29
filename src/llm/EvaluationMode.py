from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .Config import ModelConfig, TrainConfig
from .DataModule import DataModuleConfig, SequenceDataModule
from .EarlyStopping import EarlyStopping
from .EvalResult import EvalResult
from .EvaluationProbe import EvalContext
from .Evaluator import Evaluator
from .Model import TinyGPTLanguageModel


@dataclass(frozen=True)
class SampledLossEvaluator:
    """Sampled cross-entropy over a split of a pre-built Evaluator.

    Strategy (the evaluator and its RNG) lives on the instance; the per-run
    inputs (split, name, provenance) come from the ``EvalContext``.
    """
    evaluator: Evaluator
    generator: torch.Generator | None = None

    def evaluate(self, context: EvalContext) -> EvalResult:
        return self.evaluator.estimate_split_result(
            context.split,
            self.generator,
            name=context.name,
            checkpoint=context.checkpoint,
            corpus=context.corpus,
            notes=context.notes,
        )


@dataclass(frozen=True)
class FullSplitEvaluator:
    """Deterministic full-pass cross-entropy over a split.

    Strategy (the evaluator, batch size, stride) lives on the instance; the
    per-run inputs (split, name, provenance) come from the ``EvalContext``.
    """
    evaluator: Evaluator
    batch_size: int | None = None
    stride: int | None = None

    def evaluate(self, context: EvalContext) -> EvalResult:
        return self.evaluator.estimate_split_full_result(
            context.split,
            batch_size=self.batch_size,
            stride=self.stride,
            name=context.name,
            checkpoint=context.checkpoint,
            corpus=context.corpus,
            notes=context.notes,
        )


@dataclass(frozen=True)
class PerBookEvaluator:
    """Evaluate one book's token sequence, sampled or full-split.

    Strategy (model, configs, seed, full-split choice) lives on the instance;
    the book's identity and ``tokens`` come from the ``EvalContext``.
    """
    model: TinyGPTLanguageModel
    model_config: ModelConfig
    train_config: TrainConfig
    seed: int
    full_split: bool = False
    stride: int | None = None

    def evaluate(self, context: EvalContext) -> EvalResult:
        if context.tokens is None:
            raise ValueError("PerBookEvaluator requires context.tokens")
        evaluator = evaluator_for_tokens(
            self.model,
            context.tokens,
            self.model_config,
            self.train_config,
        )
        if self.full_split:
            return FullSplitEvaluator(
                evaluator,
                stride=self.stride,
            ).evaluate(context)
        return SampledLossEvaluator(
            evaluator,
            generator=book_generator(self.seed, context.name),
        ).evaluate(context)


@dataclass(frozen=True)
class CorruptionEvaluator:
    """Evaluate a book after applying a structure-destroying corruption.

    Strategy (model, configs, seed, corruption fn) lives on the instance; the
    book's identity and ``raw`` bytes come from the ``EvalContext``.
    """
    model: TinyGPTLanguageModel
    model_config: ModelConfig
    train_config: TrainConfig
    seed: int
    corrupt: Callable[[bytes, str], bytes]

    def evaluate(self, context: EvalContext) -> EvalResult:
        if context.raw is None:
            raise ValueError("CorruptionEvaluator requires context.raw")
        corrupted = self.corrupt(context.raw, context.name)
        tokens = torch.tensor(bytearray(corrupted), dtype=torch.long)
        per_book_context = EvalContext(
            name=context.name,
            split=context.split,
            tokens=tokens,
            corpus=context.corpus,
            checkpoint=context.checkpoint,
            notes=context.notes,
        )
        return PerBookEvaluator(
            self.model,
            self.model_config,
            self.train_config,
            self.seed,
        ).evaluate(per_book_context)


@dataclass(frozen=True)
class BaselineEvaluator:
    """Wrap an externally-computed loss (e.g. an n-gram baseline) as a result.

    The loss is supplied on the instance (it is computed elsewhere); the
    ``EvalContext`` provides identity and provenance. ``nTokens`` is also an
    instance field since it describes the externally-computed measurement.
    """
    loss: float
    method: str = "baseline"
    n_tokens: int | None = None

    def evaluate(self, context: EvalContext) -> EvalResult:
        return EvalResult(
            name=context.name,
            split=context.split,
            loss=self.loss,
            nTokens=self.n_tokens,
            method=self.method,
            corpus=context.corpus,
            notes=context.notes,
        )


def evaluator_for_tokens(
    model: TinyGPTLanguageModel,
    tokens: torch.Tensor,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Evaluator:
    data_module = SequenceDataModule(
        model_config,
        DataModuleConfig.fromTrainConfig(train_config),
        sequence=tokens,
        validationSequence=tokens,
    )
    return Evaluator(
        model, data_module, train_config, EarlyStopping(patience=1, delta=0.0)
    )


def book_seed(seed: int, book_name: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"{seed}:{book_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**63)


def book_generator(seed: int, book_name: str) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(book_seed(seed, book_name))
    return generator
