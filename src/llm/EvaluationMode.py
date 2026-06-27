from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .Config import ModelConfig, TrainConfig
from .DataModule import DataModuleConfig, SequenceDataModule
from .EarlyStopping import EarlyStopping
from .EvalResult import EvalResult
from .Evaluator import Evaluator
from .Model import TinyGPTLanguageModel


@dataclass(frozen=True)
class SampledLossEvaluator:
    evaluator: Evaluator
    split: str = "val"
    generator: torch.Generator | None = None
    name: str | None = None
    checkpoint: str | None = None
    corpus: str | None = None
    notes: str | None = None

    def evaluate(self) -> EvalResult:
        return self.evaluator.estimate_split_result(
            self.split,
            self.generator,
            name=self.name,
            checkpoint=self.checkpoint,
            corpus=self.corpus,
            notes=self.notes,
        )


@dataclass(frozen=True)
class FullSplitEvaluator:
    evaluator: Evaluator
    split: str = "val"
    batch_size: int | None = None
    stride: int | None = None
    name: str | None = None
    checkpoint: str | None = None
    corpus: str | None = None
    notes: str | None = None

    def evaluate(self) -> EvalResult:
        return self.evaluator.estimate_split_full_result(
            self.split,
            batch_size=self.batch_size,
            stride=self.stride,
            name=self.name,
            checkpoint=self.checkpoint,
            corpus=self.corpus,
            notes=self.notes,
        )


@dataclass(frozen=True)
class PerBookEvaluator:
    model: TinyGPTLanguageModel
    model_config: ModelConfig
    train_config: TrainConfig
    seed: int
    full_split: bool = False
    stride: int | None = None
    checkpoint: str | None = None

    def evaluate(
        self,
        book_name: str,
        tokens: torch.Tensor,
        corpus: str | None = None,
    ) -> EvalResult:
        evaluator = evaluator_for_tokens(
            self.model,
            tokens,
            self.model_config,
            self.train_config,
        )
        if self.full_split:
            return FullSplitEvaluator(
                evaluator,
                split="val",
                stride=self.stride,
                name=book_name,
                checkpoint=self.checkpoint,
                corpus=corpus,
            ).evaluate()
        return SampledLossEvaluator(
            evaluator,
            split="val",
            generator=book_generator(self.seed, book_name),
            name=book_name,
            checkpoint=self.checkpoint,
            corpus=corpus,
        ).evaluate()


@dataclass(frozen=True)
class CorruptionEvaluator:
    model: TinyGPTLanguageModel
    model_config: ModelConfig
    train_config: TrainConfig
    seed: int
    corrupt: Callable[[bytes, str], bytes]
    checkpoint: str | None = None

    def evaluate(
        self,
        name: str,
        raw: bytes,
        corpus: str | None = None,
    ) -> EvalResult:
        tokens = torch.tensor(bytearray(self.corrupt(raw, name)), dtype=torch.long)
        return PerBookEvaluator(
            self.model,
            self.model_config,
            self.train_config,
            self.seed,
            checkpoint=self.checkpoint,
        ).evaluate(name, tokens, corpus=corpus)


@dataclass(frozen=True)
class BaselineEvaluator:
    method: str = "baseline"

    def evaluate(
        self,
        name: str,
        split: str,
        loss: float,
        n_tokens: int | None = None,
        corpus: str | None = None,
        notes: str | None = None,
    ) -> EvalResult:
        return EvalResult(
            name=name,
            split=split,
            loss=loss,
            nTokens=n_tokens,
            method=self.method,
            corpus=corpus,
            notes=notes,
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
