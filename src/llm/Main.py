import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import torch

from llm.Config import RunConfig, ModelConfig, TrainConfig
from llm.DataModule import TokenDataModule, Utf8ByteTokenizer, ByteDataModule, SequenceDataModule
from llm.Model import TinyGPTLanguageModel
from llm.Trainer import LMTrainer
from llm.TextGenerator import AutoregressiveGenerator
from llm.Evaluator import Evaluator
from llm.EarlyStopping import EarlyStopping
from llm.RunArtifacts import RunArtifacts
from llm.TrainingCallback import TrainingCallback, LoggingCallback, MetricsCallback, CheckpointCallback, TrainingCurveCallback
from llm.tensor_utils import resolve_device

logger = logging.getLogger(__name__)

LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class TrainCliConfig(NamedTuple):
    runConfig: RunConfig
    resetEarlyStopping: bool
    logLevel: int


def setupLogging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(__name__)


def manual_seed(seed: int) -> torch.Generator:
    seed_fn: Callable[[int], torch.Generator] = torch.manual_seed  # type: ignore[reportUnknownMemberType]
    return seed_fn(seed)


def build_data_module(modelConfig: ModelConfig, trainConfig: TrainConfig, activeLogger: logging.Logger) -> SequenceDataModule:
    mode = trainConfig.dataModule.lower()
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

    device = resolve_device(trainConfig.device, activeLogger)
    trainConfig = replace(trainConfig, device=device)

    dataModule = build_data_module(modelConfig, trainConfig, activeLogger)

    activeLogger.info("Building model...")
    model = TinyGPTLanguageModel(modelConfig).to(trainConfig.device)

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

    runArtifacts = RunArtifacts(modelConfig, trainConfig)

    # Callbacks are registered in the order they fire on each event.
    # CheckpointCallback holds a reference to the trainer and is patched in
    # after construction to break the circular dependency.
    trainer = LMTrainer(
        modelConfig, trainConfig, model, dataModule,
        logger=activeLogger,
        evaluator=evaluator,
    )

    callbacks: list[TrainingCallback] = [
        LoggingCallback(trainConfig, activeLogger),
        MetricsCallback(runArtifacts),
        CheckpointCallback(trainer, activeLogger),
        TrainingCurveCallback(modelConfig, trainConfig, activeLogger),
    ]
    trainer.callbacks = callbacks
    trainer.runArtifacts = runArtifacts
    return trainer


def writeRunMetadata(trainer: LMTrainer, logger: logging.Logger) -> None:
    wasContinuation = trainer.runArtifacts.runMetadataExists()
    metadataPath = trainer.runArtifacts.writeRunMetadata()
    if metadataPath is not None:
        logger.info("Run metadata written to %s", metadataPath)
    if wasContinuation:
        continuationPath = trainer.runArtifacts.appendContinuation()
        if continuationPath is not None:
            logger.info("Run continuation metadata written to %s", continuationPath)


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the tiny LLM")
    parser.add_argument("--corpus", type=str, default=None, help="Path to training corpus (overrides TrainConfig.dataPath)")
    parser.add_argument("--validation-corpus", type=str, default=None, help="Path to validation corpus")
    parser.add_argument("--test-corpus", type=str, default=None, help="Path to test corpus")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to training checkpoint")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory; sets checkpoint to RUN_DIR/checkpoints/best.pt")
    parser.add_argument("--seed", type=int, default=None, help="Training random seed")
    parser.add_argument("--block-size", type=int, default=None, help="Transformer context window size")
    parser.add_argument("--n-embed", type=int, default=None, help="Transformer embedding width")
    parser.add_argument("--n-head", type=int, default=None, help="Transformer attention heads")
    parser.add_argument("--n-layer", type=int, default=None, help="Transformer decoder layers")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum optimizer steps to train")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="Evaluations without significant improvement before stopping")
    parser.add_argument("--reset-early-stopping", action="store_true", help="Reset restored early-stopping progress when resuming")
    parser.add_argument("--snapshot-interval", type=int, default=None, help="Steps between retained checkpoint snapshots")
    parser.add_argument("--max-snapshots", type=int, default=None, help="Maximum retained step snapshots")
    parser.add_argument("--plot", action="store_true", help="Enable plotting the training curve")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser


def runConfigFromArgs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    defaultLogLevel: int = logging.INFO,
) -> TrainCliConfig:
    level_name = args.log_level.upper()
    if level_name not in LOG_LEVELS:
        valid = ", ".join(LOG_LEVELS)
        parser.error(f"--log-level must be one of: {valid}")
    level = LOG_LEVELS.get(level_name, defaultLogLevel)
    runConfig = RunConfig()
    modelConfig = runConfig.modelConfig
    trainConfig = runConfig.trainConfig
    if args.corpus:
        trainConfig = replace(trainConfig, dataPath=args.corpus)
    if args.validation_corpus:
        trainConfig = replace(trainConfig, validationDataPath=args.validation_corpus)
    if args.test_corpus:
        trainConfig = replace(trainConfig, testDataPath=args.test_corpus)
    if args.checkpoint:
        trainConfig = replace(trainConfig, ckptPath=args.checkpoint)
    if args.run_dir:
        trainConfig = replace(
            trainConfig,
            ckptPath=str(Path(args.run_dir) / "checkpoints" / "best.pt"),
        )
    if args.block_size is not None:
        if args.block_size <= 0:
            parser.error("--block-size must be greater than zero")
        modelConfig = replace(modelConfig, blockSize=args.block_size)
    if args.n_embed is not None:
        if args.n_embed <= 0:
            parser.error("--n-embed must be greater than zero")
        modelConfig = replace(modelConfig, nEmbed=args.n_embed)
    if args.n_head is not None:
        if args.n_head <= 0:
            parser.error("--n-head must be greater than zero")
        modelConfig = replace(modelConfig, nHead=args.n_head)
    if args.n_layer is not None:
        if args.n_layer <= 0:
            parser.error("--n-layer must be greater than zero")
        modelConfig = replace(modelConfig, nLayer=args.n_layer)
    if args.seed is not None:
        if args.seed < 0:
            parser.error("--seed must be non-negative")
        trainConfig = replace(trainConfig, seed=args.seed)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            parser.error("--max-steps must be greater than zero")
        trainConfig = replace(trainConfig, maxSteps=args.max_steps)
    if args.early_stop_patience is not None:
        if args.early_stop_patience <= 0:
            parser.error("--early-stop-patience must be greater than zero")
        trainConfig = replace(
            trainConfig,
            earlyStopPatience=args.early_stop_patience,
        )
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
    return TrainCliConfig(
        runConfig=RunConfig(modelConfig=modelConfig, trainConfig=trainConfig),
        resetEarlyStopping=args.reset_early_stopping,
        logLevel=level,
    )


def parseTrainCli(
    argv: Sequence[str] | None = None,
    defaultLogLevel: int = logging.INFO,
) -> TrainCliConfig:
    parser = buildParser()
    args = parser.parse_args(argv)
    return runConfigFromArgs(args, parser, defaultLogLevel=defaultLogLevel)


def main(argv: Sequence[str] | int | None = None, log_level: int = logging.INFO) -> None:
    if isinstance(argv, int):
        log_level = argv
        argv = None
    cliConfig = parseTrainCli(argv, defaultLogLevel=log_level)
    activeLogger = setupLogging(level=cliConfig.logLevel)

    activeLogger.info("Building trainer...")
    trainer = buildTrainer(cliConfig.runConfig, log=activeLogger)

    activeLogger.info("Loading checkpoint (if any)...")
    trainer.loadCheckpointIfExists(
        resetEarlyStopping=cliConfig.resetEarlyStopping,
    )

    writeRunMetadata(trainer, activeLogger)

    trainingRan = trainer.train()
    if trainingRan is False:
        activeLogger.info("Skipping final test evaluation and sample save.")
        return
    trainer.evaluateBestCheckpointOnTest()

    textGenerator = AutoregressiveGenerator(trainer.model, trainer.trainConfig.device, activeLogger)
    runDirectory = trainer.trainConfig.runDirectory()
    samplePath = (
        runDirectory / "samples" / "sample.txt"
        if runDirectory is not None
        else Path("tmp") / "sample.txt"
    )
    textGenerator.saveSample(maxNewTokens=200, prompt="", path=samplePath)


if __name__ == "__main__":
    main()
