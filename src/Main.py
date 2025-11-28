from __future__ import annotations

import logging
import torch

from typing import Callable

from Config import RunConfig
from DataModule import ByteDataModule
from Model import TinyGpt
from Trainer import Trainer
from TextGenerator import TextGenerator


logger = logging.getLogger(__name__)


def setupLogging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def manual_seed(seed: int) -> torch.Generator:
    seed_fn: Callable[[int], torch.Generator] = torch.manual_seed  # type: ignore[reportUnknownMemberType]
    return seed_fn(seed)


def buildTrainer(
    runConfig : RunConfig | None = None,
    log: logging.Logger | None = None,
) -> Trainer:
    manual_seed(1337)

    runConfig  = runConfig  or RunConfig()
    modelConfig = runConfig .model
    train_cfg = runConfig .train

    activeLogger = log or logger

    activeLogger.info("Loading data module...")
    dataModule = ByteDataModule(modelConfig, train_cfg)

    activeLogger.info("Building model...")
    model = TinyGpt(modelConfig).to(train_cfg.device)

    return Trainer(modelConfig, train_cfg, model, dataModule)


def main(log_level: int = logging.INFO) -> None:
    activeLogger = setupLogging(level=log_level)
    activeLogger.info("Building trainer...")
    trainer = buildTrainer(log=activeLogger)

    activeLogger.info("Loading checkpoint (if any)...")
    trainer.loadCheckpointIfExists()

    trainer.train()
    trainer.plotTrainingCurve()

    # Sampling handled by TextGenerator
    textGenerator = TextGenerator(trainer.model, trainer.trainCfg.device, activeLogger)
    textGenerator.log_sample(maxNewTokens=200, prompt="")


if __name__ == "__main__":
    main()
