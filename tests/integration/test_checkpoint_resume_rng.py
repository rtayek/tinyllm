from pathlib import Path
import torch
import logging

from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import ByteDataModule
from llm.Model import TinyGPTLanguageModel
from llm.Trainer import LMTrainer
from llm.Evaluator import Evaluator
from llm.EarlyStopping import EarlyStopping
from llm.TrainingCallback import CheckpointCallback
from llm.train_app import buildCheckpointContext


def test_checkpoint_restores_generator_state(tmp_path: Path) -> None:
    dataPath = tmp_path / "input.txt"
    dataPath.write_bytes(b"hello tiny llm\n" * 50)

    ckptPath = tmp_path / "ckpt.pt"

    modelConfig = ModelConfig(blockSize=8, vocabSize=256, nEmbed=16, nHead=2, nLayer=1, dropout=0.0)
    trainConfig = TrainConfig(
        batchSize=2,
        learningRate=1e-3,
        warmupFrac=0.1,
        maxSteps=3,
        evalInterval=1,
        evalIters=2,
        weightDecay=0.0,
        plotCurve=False,
        ckptPath=str(ckptPath),
        dataPath=str(dataPath),
        device="cpu",
    )

    torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
    dataModule = ByteDataModule(modelConfig, trainConfig)
    model = TinyGPTLanguageModel(modelConfig).to(trainConfig.device)

    mock_logger = logging.getLogger("test_logger")
    mock_early_stopping = EarlyStopping(patience=1, delta=0.0)
    evaluator = Evaluator(
        model=model,
        data_module=dataModule,
        trainConfig=trainConfig,
        early_stopping=mock_early_stopping,
        logger=mock_logger,
    )
    trainer = LMTrainer(modelConfig, trainConfig, model, dataModule, evaluator=evaluator, logger=mock_logger)
    trainer.callbacks = [
        CheckpointCallback(buildCheckpointContext(trainer), mock_logger)
    ]

    trainer.loadCheckpointIfExists()
    trainer.train()

    assert ckptPath.exists(), "Checkpoint file should be written"

    latestPath = ckptPath.with_name("latest.pt")
    assert latestPath.exists(), "Latest checkpoint should be written"
    checkpoint = torch.load(latestPath, map_location=trainConfig.device, weights_only=False)  # pyright: ignore[reportUnknownMemberType]
    generatorState = checkpoint.get("generatorState", None)
    evaluatorGeneratorState = checkpoint.get("evaluatorGeneratorState", None)
    earlyStoppingState = checkpoint.get("earlyStoppingState", None)
    assert generatorState is not None
    assert evaluatorGeneratorState is not None
    assert earlyStoppingState is not None

    dataModuleTwo = ByteDataModule(modelConfig, trainConfig)
    modelTwo = TinyGPTLanguageModel(modelConfig).to(trainConfig.device)
    evaluatorTwo = Evaluator(
        model=modelTwo,
        data_module=dataModuleTwo,
        trainConfig=trainConfig,
        early_stopping=EarlyStopping(patience=1, delta=0.0),
        logger=mock_logger,
    )
    evaluatorTwo.early_stopping.noImproveEvals = 2
    trainerTwo = LMTrainer(modelConfig, trainConfig, modelTwo, dataModuleTwo, evaluator=evaluatorTwo, logger=mock_logger)
    trainerTwo.loadCheckpointIfExists()

    assert torch.equal(trainerTwo.generator.get_state(), generatorState)
    assert torch.equal(
        evaluatorTwo.generator.get_state(),
        evaluatorGeneratorState,
    )
    assert evaluatorTwo.early_stopping.noImproveEvals == earlyStoppingState[
        "noImproveEvals"
    ]

    genCopy = torch.Generator()
    genCopy.set_state(generatorState)
    batchXExpected, _ = dataModuleTwo.getBatch("train", genCopy)
    batchXActual, _ = dataModuleTwo.getBatch("train", trainerTwo.generator)
    assert torch.equal(batchXExpected, batchXActual)
