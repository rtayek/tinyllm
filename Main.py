# Main.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
import torch

from Config import ModelConfig, TrainConfig
from DataModule import ByteDataModule
from Model import TinyGpt
from Trainer import Trainer

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def build_trainer(
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    log: logging.Logger | None = None,
) -> Trainer:
    torch.manual_seed(1337)

    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    active_logger = log or logger

    active_logger.info("Loading data module...")
    data_module = ByteDataModule(model_cfg, train_cfg)

    active_logger.info("Building model...")
    model = TinyGpt(model_cfg).to(train_cfg.device)

    return Trainer(model_cfg, train_cfg, model, data_module)


def main() -> None:
    active_logger = setup_logging()
    trainer = build_trainer(log=active_logger)

    active_logger.info("Loading checkpoint (if any)...")
    trainer.loadCheckpointIfExists()

    trainer.train()
    trainer.plotTrainingCurve()
    trainer.printSample(maxNewTokens=200)


if __name__ == "__main__":
    main()
