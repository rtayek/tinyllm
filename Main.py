# Main.py

from __future__ import annotations

import logging
import torch

from Config import RunConfig
from DataModule import ByteDataModule
from Model import TinyGpt
from Trainer import Trainer
from typing import Callable

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def manual_seed(seed: int) -> torch.Generator:
    seed_fn: Callable[[int], torch.Generator] = torch.manual_seed  # type: ignore[reportUnknownMemberType]
    return seed_fn(seed)


def build_trainer(
    run_cfg: RunConfig | None = None,
    log: logging.Logger | None = None,
) -> Trainer:
    manual_seed(1337)

    run_cfg = run_cfg or RunConfig()
    model_cfg = run_cfg.model
    train_cfg = run_cfg.train

    active_logger = log or logger

    active_logger.info("Loading data module...")
    data_module = ByteDataModule(model_cfg, train_cfg)

    active_logger.info("Building model...")
    model = TinyGpt(model_cfg).to(train_cfg.device)

    return Trainer(model_cfg, train_cfg, model, data_module)


def main(log_level: int = logging.INFO) -> None:
    active_logger = setup_logging(level=log_level)
    trainer = build_trainer(log=active_logger)

    active_logger.info("Loading checkpoint (if any)...")
    trainer.loadCheckpointIfExists()

    trainer.train()
    trainer.plotTrainingCurve()
    trainer.printSample(maxNewTokens=200)


if __name__ == "__main__":
    main()
