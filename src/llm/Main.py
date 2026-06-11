import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable

import torch

from llm.Config import RunConfig, ModelConfig, TrainConfig
from llm.DataModule import TokenDataModule, Utf8ByteTokenizer, ByteDataModule, SequenceDataModule
from llm.Model import TinyGPTLanguageModel
from llm.Trainer import LMTrainer
from llm.TextGenerator import AutoregressiveGenerator
from llm.Evaluator import Evaluator
from llm.EarlyStopping import EarlyStopping
from llm.tensor_utils import resolve_device

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


def buildTrainer(runConfig: RunConfig | None = None, log: logging.Logger | None = None) -> LMTrainer:
    runConfig = runConfig or RunConfig()
    modelConfig = runConfig.modelConfig
    trainConfig = runConfig.trainConfig
    manual_seed(trainConfig.seed)

    activeLogger = log or logger

    # 🔍 Reconcile desired device vs actual availability
    device = resolve_device(trainConfig.device, activeLogger)
    trainConfig = replace(trainConfig, device=device)

    dataModule = build_data_module(modelConfig, trainConfig, activeLogger)

    activeLogger.info("Building model...")
    model = TinyGPTLanguageModel(modelConfig).to(trainConfig.device)

    # Instantiate EarlyStopping and Evaluator
    earlyStopping = EarlyStopping(trainConfig.earlyStopPatience, trainConfig.earlyStopDelta)
    evaluatorGenerator = torch.Generator()
    evaluatorGenerator.manual_seed(trainConfig.seed + 1)
    evaluator = Evaluator(
        model,
        dataModule,
        trainConfig,
        earlyStopping,
        generator=evaluatorGenerator,
        logger=activeLogger,
    )

    return LMTrainer(modelConfig, trainConfig, model, dataModule, logger=activeLogger, evaluator=evaluator)


def main(log_level: int = logging.INFO) -> None:
    parser = argparse.ArgumentParser(description="Train the tiny LLM")
    parser.add_argument("--corpus", type=str, default=None, help="Path to training corpus (overrides TrainConfig.dataPath)")
    parser.add_argument("--validation-corpus", type=str, default=None, help="Path to validation corpus")
    parser.add_argument("--test-corpus", type=str, default=None, help="Path to test corpus")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to training checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Training random seed")
    parser.add_argument("--snapshot-interval", type=int, default=None, help="Steps between retained checkpoint snapshots")
    parser.add_argument("--max-snapshots", type=int, default=None, help="Maximum retained step snapshots")
    parser.add_argument("--plot", action="store_true", help="Enable plotting the training curve")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), log_level)
    activeLogger = setupLogging(level=level)

    runConfig = RunConfig()
    trainConfig = runConfig.trainConfig
    if args.corpus:
        trainConfig = replace(trainConfig, dataPath=args.corpus)
    if args.validation_corpus:
        trainConfig = replace(trainConfig, validationDataPath=args.validation_corpus)
    if args.test_corpus:
        trainConfig = replace(trainConfig, testDataPath=args.test_corpus)
    if args.checkpoint:
        trainConfig = replace(trainConfig, ckptPath=args.checkpoint)
    if args.seed is not None:
        if args.seed < 0:
            parser.error("--seed must be non-negative")
        trainConfig = replace(trainConfig, seed=args.seed)
    if args.snapshot_interval is not None:
        if args.snapshot_interval < 0:
            parser.error("--snapshot-interval must be non-negative")
        trainConfig = replace(trainConfig, snapshotInterval=args.snapshot_interval)
    if args.max_snapshots is not None:
        if args.max_snapshots < 0:
            parser.error("--max-snapshots must be non-negative")
        trainConfig = replace(trainConfig, maxSnapshots=args.max_snapshots)
    if args.plot:
        trainConfig = replace(trainConfig, plotCurve=True)
    runConfig = RunConfig(modelConfig=runConfig.modelConfig, trainConfig=trainConfig)

    activeLogger.info("Building trainer...")
    trainer = buildTrainer(runConfig, log=activeLogger)

    activeLogger.info("Loading checkpoint (if any)...")
    trainer.loadCheckpointIfExists()

    trainer.train()
    trainer.evaluateBestCheckpointOnTest()
    trainer.plotTrainingCurve()

    textGenerator = AutoregressiveGenerator(trainer.model, trainer.trainConfig.device, activeLogger)
    runDirectory = trainer.trainConfig.runDirectory()
    samplePath = (
        runDirectory / "samples" / "sample.txt"
        if runDirectory is not None
        else Path("tmp") / "sample.txt"
    )
    textGenerator.saveSample(maxNewTokens=200, prompt="", path=samplePath)
    #textGenerator.logSample(maxNewTokens=200, prompt="")


if __name__ == "__main__":
    main()
