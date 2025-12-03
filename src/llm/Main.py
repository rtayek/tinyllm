import argparse
import logging
from dataclasses import replace
from typing import Callable

import torch

from llm.Config import RunConfig, ModelConfig, TrainConfig
from llm.DataModule import TokenDataModule, Utf8ByteTokenizer, ByteDataModule, SequenceDataModule
from llm.Model import TinyGpt
from llm.Trainer import Trainer
from llm.TextGenerator import TextGenerator
from llm.Evaluator import Evaluator
from llm.EarlyStopping import EarlyStopping

logger = logging.getLogger(__name__)


def setupLogging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(__name__)


def manual_seed(seed: int) -> torch.Generator:
    seed_fn: Callable[[int], torch.Generator] = torch.manual_seed  # type: ignore[reportUnknownMemberType]
    return seed_fn(seed)


def build_data_module(modelConfig: ModelConfig, trainConfig: TrainConfig, activeLogger: logging.Logger) -> SequenceDataModule:
    mode = getattr(trainConfig, "dataModule", "token").lower()
    if mode in ("byte", "bytes"):
        activeLogger.info("Loading data module: raw bytes")
        return ByteDataModule(modelConfig, trainConfig, logger=activeLogger)

    if mode != "token":
        raise ValueError(f"Unknown dataModule '{mode}'; expected 'token' or 'byte'")

    activeLogger.info("Loading data module: tokenized UTF-8 bytes")
    tokenizer = Utf8ByteTokenizer()
    if modelConfig.vocabSize != tokenizer.vocabSize:
        activeLogger.warning("modelConfig.vocabSize (%s) differs from tokenizer vocabSize (%s)", modelConfig.vocabSize, tokenizer.vocabSize)
    return TokenDataModule(modelConfig, trainConfig, tokenizer=tokenizer, logger=activeLogger)


def buildTrainer(runConfig: RunConfig | None = None, log: logging.Logger | None = None) -> Trainer:
    manual_seed(1337)

    runConfig = runConfig or RunConfig()
    modelConfig = runConfig.modelConfig
    trainConfig = runConfig.trainConfig

    activeLogger = log or logger

    # 🔍 Reconcile desired device vs actual availability
    if trainConfig.device == "cuda" and not torch.cuda.is_available():
        activeLogger.warning("CUDA requested in TrainConfig, but not available; falling back to cpu")
        trainConfig = replace(trainConfig, device="cpu")
        runConfig = RunConfig(modelConfig=modelConfig, trainConfig=trainConfig)

    dataModule = build_data_module(modelConfig, trainConfig, activeLogger)

    activeLogger.info("Building model...")
    model = TinyGpt(modelConfig).to(trainConfig.device)

    # Instantiate EarlyStopping and Evaluator
    earlyStopping = EarlyStopping(trainConfig.earlyStopPatience, trainConfig.earlyStopDelta)
    evaluator = Evaluator(model, dataModule, trainConfig, earlyStopping, logger=activeLogger)

    return Trainer(modelConfig, trainConfig, model, dataModule, logger=activeLogger, evaluator=evaluator)


def main(log_level: int = logging.INFO) -> None:
    parser = argparse.ArgumentParser(description="Train the tiny LLM")
    parser.add_argument("--corpus", type=str, default=None, help="Path to training corpus (overrides TrainConfig.dataPath)")
    parser.add_argument("--plot", action="store_true", help="Enable plotting the training curve")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), log_level)
    activeLogger = setupLogging(level=level)

    runConfig = RunConfig()
    trainConfig = runConfig.trainConfig
    if args.corpus:
        trainConfig = replace(trainConfig, dataPath=args.corpus)
    if args.plot:
        trainConfig = replace(trainConfig, plotCurve=True)
    runConfig = RunConfig(modelConfig=runConfig.modelConfig, trainConfig=trainConfig)

    activeLogger.info("Building trainer...")
    trainer = buildTrainer(runConfig, log=activeLogger)

    activeLogger.info("Loading checkpoint (if any)...")
    trainer.loadCheckpointIfExists()

    trainer.train()
    trainer.plotTrainingCurve()

    textGenerator = TextGenerator(trainer.model, trainer.trainConfig.device, activeLogger)
    textGenerator.log_sample(maxNewTokens=200, prompt="")


if __name__ == "__main__":
    main()
