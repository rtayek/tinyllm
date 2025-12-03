from pathlib import Path
import torch
import logging

from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import ByteDataModule
from llm.Model import TinyGpt
from llm.Trainer import Trainer
from llm.Evaluator import Evaluator
from llm.EarlyStopping import EarlyStopping


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
    model = TinyGpt(modelConfig).to(trainConfig.device)

    mock_logger = logging.getLogger("test_logger")
    mock_early_stopping = EarlyStopping(patience=1, delta=0.0)
    evaluator = Evaluator(
        model=model,
        data_module=dataModule,
        trainConfig=trainConfig,
        early_stopping=mock_early_stopping,
        logger=mock_logger,
    )
    trainer = Trainer(modelConfig, trainConfig, model, dataModule, evaluator=evaluator, logger=mock_logger)

    trainer.loadCheckpointIfExists()
    trainer.train()

    assert ckptPath.exists(), "Checkpoint file should be written"

    checkpoint = torch.load(ckptPath, map_location=trainConfig.device, weights_only=False)  # pyright: ignore[reportUnknownMemberType]
    generatorState = checkpoint.get("generatorState", None)
    assert generatorState is not None

    dataModuleTwo = ByteDataModule(modelConfig, trainConfig)
    modelTwo = TinyGpt(modelConfig).to(trainConfig.device)
    evaluatorTwo = Evaluator(
        model=modelTwo,
        data_module=dataModuleTwo,
        trainConfig=trainConfig,
        early_stopping=mock_early_stopping, # Use the same mock early stopping for simplicity
        logger=mock_logger,
    )
    trainerTwo = Trainer(modelConfig, trainConfig, modelTwo, dataModuleTwo, evaluator=evaluatorTwo, logger=mock_logger)
    trainerTwo.loadCheckpointIfExists()

    assert torch.equal(trainerTwo.generator.get_state(), generatorState)

    genCopy = torch.Generator()
    genCopy.set_state(generatorState)
    batchXExpected, _ = dataModuleTwo.getBatch("train", genCopy)
    batchXActual, _ = dataModuleTwo.getBatch("train", trainerTwo.generator)
    assert torch.equal(batchXExpected, batchXActual)
