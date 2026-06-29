from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..Checkpoint import Checkpoint
from ..Config import ModelConfig, TrainConfig
from ..EvalResult import EvalResult
from ..EvaluationProbe import EvalContext
from ..EvaluationMode import (
    PerBookEvaluator,
    book_generator as _book_generator,
    book_seed as _book_seed,
    evaluator_for_tokens as _evaluator_for_tokens,
)
from ..Evaluator import Evaluator
from ..Model import TinyGPTLanguageModel
from ..research_books import RESEARCH_BOOKS


@dataclass(frozen=True)
class ResearchBookData:
    name: str
    path: Path
    raw: bytes
    tokens: torch.Tensor


@dataclass(frozen=True)
class CheckpointModel:
    model_config: ModelConfig
    train_config: TrainConfig
    model: TinyGPTLanguageModel


def load_tokens(path: Path) -> torch.Tensor:
    return torch.tensor(bytearray(path.read_bytes()), dtype=torch.long)


def book_seed(seed: int, book_name: str) -> int:
    return _book_seed(seed, book_name)


def book_generator(seed: int, book_name: str) -> torch.Generator:
    return _book_generator(seed, book_name)


def load_checkpoint_model(
    checkpoint_path: str,
    device: str,
    eval_iters: int,
) -> CheckpointModel:
    checkpoint = Checkpoint.load(checkpoint_path, device)
    if not checkpoint.modelConfig:
        raise ValueError("checkpoint has no modelConfig")

    model_config = ModelConfig.fromDict(checkpoint.modelConfig)
    train_config = TrainConfig(device=device, evalIters=eval_iters)
    model = TinyGPTLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint.modelState)
    model.eval()
    return CheckpointModel(model_config, train_config, model)


def evaluator_for_tokens(
    model: TinyGPTLanguageModel,
    tokens: torch.Tensor,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Evaluator:
    return _evaluator_for_tokens(model, tokens, model_config, train_config)


def estimate_validation_loss(
    model: TinyGPTLanguageModel,
    tokens: torch.Tensor,
    model_config: ModelConfig,
    train_config: TrainConfig,
    seed: int,
    book_name: str,
    full_split: bool = False,
    stride: int | None = None,
) -> float:
    return estimate_validation_result(
        model,
        tokens,
        model_config,
        train_config,
        seed,
        book_name,
        full_split=full_split,
        stride=stride,
    ).loss


def estimate_validation_result(
    model: TinyGPTLanguageModel,
    tokens: torch.Tensor,
    model_config: ModelConfig,
    train_config: TrainConfig,
    seed: int,
    book_name: str,
    full_split: bool = False,
    stride: int | None = None,
    checkpoint: str | None = None,
    corpus: str | None = None,
) -> EvalResult:
    return PerBookEvaluator(
        model,
        model_config,
        train_config,
        seed,
        full_split=full_split,
        stride=stride,
    ).evaluate(
        EvalContext(
            name=book_name,
            tokens=tokens,
            corpus=corpus,
            checkpoint=checkpoint,
        )
    )


def load_research_book_data() -> tuple[list[ResearchBookData], list[tuple[str, Path]]]:
    books: list[ResearchBookData] = []
    missing: list[tuple[str, Path]] = []
    for book in RESEARCH_BOOKS:
        path = book.validation_path
        if not path.exists():
            missing.append((book.name, path))
            continue
        raw = path.read_bytes()
        books.append(
            ResearchBookData(
                name=book.name,
                path=path,
                raw=raw,
                tokens=torch.tensor(bytearray(raw), dtype=torch.long),
            )
        )
    return books, missing
