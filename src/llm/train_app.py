import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

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
from llm.train_cli import parseTrainCli

logger = logging.getLogger(__name__)


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


def sample_path_for(trainer: LMTrainer) -> Path:
    runDirectory = trainer.trainConfig.runDirectory()
    if runDirectory is not None:
        return runDirectory / "samples" / "sample.txt"
    return Path("tmp") / "sample.txt"


def finish_training(trainer: LMTrainer, logger: logging.Logger) -> bool:
    trainingRan = trainer.train()
    if trainingRan is False:
        logger.info("Skipping final test evaluation and sample save.")
        return False

    trainer.evaluateBestCheckpointOnTest()
    textGenerator = AutoregressiveGenerator(
        trainer.model,
        trainer.trainConfig.device,
        logger,
    )
    textGenerator.saveSample(
        maxNewTokens=200,
        prompt="",
        path=sample_path_for(trainer),
    )
    return True


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
    finish_training(trainer, activeLogger)


if __name__ == "__main__":
    main()
