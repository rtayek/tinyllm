from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from .Checkpoint import Checkpoint
from .Config import ModelConfig, TrainConfig
from .DataModule import SequenceDataModule
from .EarlyStopping import EarlyStopping
from .Evaluator import Evaluator
from .Model import TinyGPTLanguageModel
from .research_books import RESEARCH_BOOKS


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
    digest = hashlib.sha256(f"{seed}:{book_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**63)


def book_generator(seed: int, book_name: str) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(book_seed(seed, book_name))
    return generator


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
    data_module = SequenceDataModule(
        model_config,
        train_config,
        sequence=tokens,
        validationSequence=tokens,
    )
    return Evaluator(
        model, data_module, train_config, EarlyStopping(patience=1, delta=0.0)
    )


def estimate_validation_loss(
    model: TinyGPTLanguageModel,
    tokens: torch.Tensor,
    model_config: ModelConfig,
    train_config: TrainConfig,
    seed: int,
    book_name: str,
    full_split: bool = False,
) -> float:
    evaluator = evaluator_for_tokens(model, tokens, model_config, train_config)
    if full_split:
        return evaluator.estimate_split_full("val")
    return evaluator.estimate_split("val", book_generator(seed, book_name))


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
