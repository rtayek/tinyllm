from pathlib import Path
import torch
import logging

from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import ByteDataModule
from llm.Model import TinyGpt
from llm.Trainer import Trainer
from llm.Evaluator import Evaluator
from llm.EarlyStopping import EarlyStopping


def test_training_smoke(tmp_path: Path) -> None:
    dataPath = tmp_path / "input.txt"
    dataPath.write_bytes(b"hello tiny llm\n" * 200)

    ckptPath = tmp_path / "ckpt.pt"

    modelConfig = ModelConfig(
        blockSize=8,
        vocabSize=256,
        nEmbed=16,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    trainConfig = TrainConfig(
        batchSize=2,
        learningRate=1e-3,
        warmupFrac=0.1,
        maxSteps=5,
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

    assert trainer.trainingCurve, "Training curve should not be empty after training"
    assert trainer.bestValLoss is not None
    assert ckptPath.exists(), "Checkpoint file should be written"
