# Main.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
import torch

from Config import ModelConfig, TrainConfig
from DataModule import ByteDataModule
from Model import TinyGpt
from Trainer import Trainer


def build_trainer(
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> Trainer:
    torch.manual_seed(1337)

    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    logger.info("Loading data module...")
    data_module = ByteDataModule(model_cfg, train_cfg)

    logger.info("Building model...")
    model = TinyGpt(model_cfg).to(train_cfg.device)

    return Trainer(model_cfg, train_cfg, model, data_module)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    trainer = build_trainer()

    logger.info("Loading checkpoint (if any)...")
    trainer.loadCheckpointIfExists()

    trainer.train()
    trainer.plotTrainingCurve()
    trainer.printSample(maxNewTokens=200)


if __name__ == "__main__":
    main()
