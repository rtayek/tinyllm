import logging
from pathlib import Path

import pytest
import torch

from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import ByteDataModule, SequenceDataModule
from llm.Model import TinyGPTLanguageModel
from llm.Trainer import LMTrainer
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
        snapshotInterval=1,
        maxSnapshots=2,
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
    mock_early_stopping = EarlyStopping(patience=100, delta=0.0)
    evaluator = Evaluator(
        model=model,
        data_module=dataModule,
        trainConfig=trainConfig,
        early_stopping=mock_early_stopping,
        logger=mock_logger,
    )
    trainer = LMTrainer(modelConfig, trainConfig, model, dataModule, evaluator=evaluator, logger=mock_logger)

    trainer.loadCheckpointIfExists()
    trainer.train()

    assert trainer.trainingCurve, "Training curve should not be empty after training"
    assert trainer.bestValLoss is not None
    assert ckptPath.exists(), "Checkpoint file should be written"
    assert ckptPath.with_name("latest.pt").exists()
    assert [path.name for path in sorted(tmp_path.glob("step-*.pt"))] == [
        "step-000003.pt",
        "step-000004.pt",
    ]


def test_missing_checkpoint_is_reported_as_new_run(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ckptPath = tmp_path / "missing.pt"
    modelConfig = ModelConfig(
        vocabSize=256,
        blockSize=4,
        nEmbed=8,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    trainConfig = TrainConfig(
        batchSize=2,
        maxSteps=1,
        evalInterval=1,
        evalIters=1,
        plotCurve=False,
        ckptPath=str(ckptPath),
        dataPath=str(tmp_path / "unused.txt"),
        validationDataPath=None,
        testDataPath=None,
        device="cpu",
    )
    sequence = torch.arange(40, dtype=torch.long)
    dataModule = SequenceDataModule(modelConfig, trainConfig, sequence)
    model = TinyGPTLanguageModel(modelConfig)
    trainer = LMTrainer(
        modelConfig,
        trainConfig,
        model,
        dataModule,
        logger=logging.getLogger("test.missing-checkpoint"),
    )

    with caplog.at_level(logging.INFO):
        trainer.loadCheckpointIfExists()

    assert "starting a new run" in caplog.text
    assert "Loaded checkpoint version" not in caplog.text
