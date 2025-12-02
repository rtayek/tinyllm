from __future__ import annotations

import logging
from typing import Callable

import torch

from llm.Config import RunConfig, ModelConfig, TrainConfig
from llm.DataModule import TokenDataModule, Utf8ByteTokenizer, ByteDataModule, SequenceDataModule
from llm.Model import TinyGpt
from llm.Trainer import Trainer
from llm.TextGenerator import TextGenerator


logger = logging.getLogger(__name__)


def setupLogging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(__name__)


def manual_seed(seed: int) -> torch.Generator:
    seed_fn: Callable[[int], torch.Generator] = torch.manual_seed  # type: ignore[reportUnknownMemberType]
    return seed_fn(seed)


def build_data_module(modelConfig: ModelConfig, train_cfg: TrainConfig, activeLogger: logging.Logger) -> SequenceDataModule:
    mode = getattr(train_cfg, "dataModule", "token").lower()
    if mode in ("byte", "bytes"):
        activeLogger.info("Loading data module: raw bytes")
        return ByteDataModule(modelConfig, train_cfg, logger=activeLogger)

    if mode != "token":
        raise ValueError(f"Unknown dataModule '{mode}'; expected 'token' or 'byte'")

    activeLogger.info("Loading data module: tokenized UTF-8 bytes")
    tokenizer = Utf8ByteTokenizer()
    if modelConfig.vocabSize != tokenizer.vocabSize:
        activeLogger.warning("modelConfig.vocabSize (%s) differs from tokenizer vocabSize (%s)", modelConfig.vocabSize, tokenizer.vocabSize)
    return TokenDataModule(modelConfig, train_cfg, tokenizer=tokenizer, logger=activeLogger)


def buildTrainer(runConfig: RunConfig | None = None, log: logging.Logger | None = None) -> Trainer:
    manual_seed(1337)

    runConfig = runConfig or RunConfig()
    modelConfig = runConfig.modelConfig
    train_cfg = runConfig.trainConfig

    activeLogger = log or logger

    dataModule = build_data_module(modelConfig, train_cfg, activeLogger)

    activeLogger.info("Building model...")
    model = TinyGpt(modelConfig).to(train_cfg.device)

    return Trainer(modelConfig, train_cfg, model, dataModule, logger=activeLogger)


def main(log_level: int = logging.INFO) -> None:
    activeLogger = setupLogging(level=log_level)
    activeLogger.info("Building trainer...")
    trainer = buildTrainer(log=activeLogger)

    activeLogger.info("Loading checkpoint (if any)...")
    trainer.loadCheckpointIfExists()

    trainer.train()
    trainer.plotTrainingCurve()

    textGenerator = TextGenerator(trainer.model, trainer.trainCfg.device, activeLogger)
    textGenerator.log_sample(maxNewTokens=200, prompt="")


if __name__ == "__main__":
    main()
