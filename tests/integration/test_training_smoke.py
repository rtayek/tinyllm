import logging
from pathlib import Path

import pytest
import torch

from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import ByteDataModule, DataModuleConfig, SequenceDataModule
from llm.Model import TinyGPTLanguageModel
from llm.Trainer import LMTrainer
from llm.Evaluator import Evaluator
from llm.EarlyStopping import EarlyStopping
from llm.Checkpoint import Checkpoint, CheckpointManager
from llm.TrainingCallback import CheckpointCallback
from llm.train_app import buildCheckpointContext


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
    trainer.callbacks = [
        CheckpointCallback(buildCheckpointContext(trainer), mock_logger)
    ]

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


def test_short_training_resumes_from_latest_checkpoint(tmp_path: Path) -> None:
    data_path = tmp_path / "input.txt"
    data_path.write_bytes(b"hello tiny resume training\n" * 100)
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    model_config = ModelConfig(
        blockSize=8,
        vocabSize=256,
        nEmbed=16,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )

    def make_trainer(max_steps: int) -> LMTrainer:
        train_config = TrainConfig(
            batchSize=2,
            learningRate=1e-3,
            warmupFrac=0.1,
            maxSteps=max_steps,
            evalInterval=1,
            evalIters=1,
            snapshotInterval=0,
            weightDecay=0.0,
            earlyStopPatience=100,
            plotCurve=False,
            ckptPath=str(checkpoint),
            dataPath=str(data_path),
            device="cpu",
        )
        data_module = ByteDataModule(model_config, train_config)
        model = TinyGPTLanguageModel(model_config).to(train_config.device)
        evaluator = Evaluator(
            model=model,
            data_module=data_module,
            trainConfig=train_config,
            early_stopping=EarlyStopping(patience=100, delta=0.0),
            logger=logging.getLogger("test.resume"),
        )
        trainer = LMTrainer(
            model_config,
            train_config,
            model,
            data_module,
            evaluator=evaluator,
            logger=logging.getLogger("test.resume"),
        )
        trainer.callbacks = [
            CheckpointCallback(
                buildCheckpointContext(trainer),
                logging.getLogger("test.resume"),
            )
        ]
        return trainer

    torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
    first = make_trainer(max_steps=2)
    first.loadCheckpointIfExists()
    first.train()
    first_latest = Checkpoint.load(str(checkpoint.with_name("latest.pt")), "cpu")
    assert first_latest.step == 2

    resumed = make_trainer(max_steps=4)
    resumed.loadCheckpointIfExists()
    assert resumed.globalStep == 2

    resumed.train()

    second_latest = Checkpoint.load(str(checkpoint.with_name("latest.pt")), "cpu")
    assert second_latest.step == 4
    assert resumed.trainingCurve[0][0] == 3


def test_completed_max_steps_run_does_not_train_again(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "input.txt"
    data_path.write_bytes(b"hello tiny completed training\n" * 100)
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    model_config = ModelConfig(
        blockSize=8,
        vocabSize=256,
        nEmbed=16,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    train_config = TrainConfig(
        batchSize=2,
        learningRate=1e-3,
        warmupFrac=0.1,
        maxSteps=3,
        evalInterval=1,
        evalIters=1,
        snapshotInterval=0,
        weightDecay=0.0,
        earlyStopPatience=100,
        plotCurve=False,
        ckptPath=str(checkpoint),
        dataPath=str(data_path),
        device="cpu",
    )

    def make_trainer() -> LMTrainer:
        data_module = ByteDataModule(model_config, train_config)
        model = TinyGPTLanguageModel(model_config).to(train_config.device)
        evaluator = Evaluator(
            model=model,
            data_module=data_module,
            trainConfig=train_config,
            early_stopping=EarlyStopping(patience=100, delta=0.0),
            logger=logging.getLogger("test.max-steps"),
        )
        trainer = LMTrainer(
            model_config,
            train_config,
            model,
            data_module,
            evaluator=evaluator,
            logger=logging.getLogger("test.max-steps"),
        )
        trainer.callbacks = [
            CheckpointCallback(
                buildCheckpointContext(trainer),
                logging.getLogger("test.max-steps"),
            )
        ]
        return trainer

    first = make_trainer()
    assert first.train() is True
    latest = Checkpoint.load(str(checkpoint.with_name("latest.pt")), "cpu")
    assert latest.step == train_config.maxSteps

    resumed = make_trainer()
    resumed.loadCheckpointIfExists()
    assert resumed.globalStep == train_config.maxSteps

    def fail_train() -> float:
        raise AssertionError("max-step-complete run should not train")

    monkeypatch.setattr(resumed, "_trainStep", fail_train)
    assert resumed.train() is False
    assert resumed.trainingCurve == []


def test_resume_rejects_incompatible_model_config(tmp_path: Path) -> None:
    data_path = tmp_path / "input.txt"
    data_path.write_bytes(b"hello tiny resume training\n" * 100)
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    saved_model_config = ModelConfig(
        blockSize=8,
        vocabSize=256,
        nEmbed=16,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    requested_model_config = ModelConfig(
        blockSize=16,
        vocabSize=256,
        nEmbed=16,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    train_config = TrainConfig(
        batchSize=2,
        maxSteps=1,
        evalInterval=1,
        evalIters=1,
        plotCurve=False,
        ckptPath=str(checkpoint),
        dataPath=str(data_path),
        validationDataPath=None,
        testDataPath=None,
        device="cpu",
    )
    data_module = ByteDataModule(saved_model_config, train_config)
    saved_model = TinyGPTLanguageModel(saved_model_config)
    optimizer = torch.optim.AdamW(saved_model.parameters())
    saved_manager = CheckpointManager(saved_model_config, train_config)
    saved_manager.saveCheckpoint(
        saved_model,
        optimizer,
        lrStrategyState=None,
        step=0,
        bestValLoss=None,
        path=str(checkpoint.with_name("latest.pt")),
    )

    requested_model = TinyGPTLanguageModel(requested_model_config)
    evaluator = Evaluator(
        requested_model,
        data_module,
        train_config,
        EarlyStopping(patience=2, delta=0.0),
    )
    trainer = LMTrainer(
        requested_model_config,
        train_config,
        requested_model,
        data_module,
        evaluator=evaluator,
    )

    with pytest.raises(ValueError, match="Checkpoint model config is incompatible"):
        trainer.loadCheckpointIfExists()


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
    dataModule = SequenceDataModule(
        modelConfig,
        DataModuleConfig.fromTrainConfig(trainConfig),
        sequence,
    )
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


def test_best_checkpoint_is_evaluated_on_test_split(tmp_path: Path) -> None:
    run_directory = tmp_path / "runs" / "test-evaluation"
    checkpoint = run_directory / "checkpoints" / "best.pt"
    model_config = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=8,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    train_config = TrainConfig(
        batchSize=2,
        evalIters=2,
        ckptPath=str(checkpoint),
        validationDataPath=None,
        testDataPath=None,
        device="cpu",
    )
    data_module = SequenceDataModule(
        model_config,
        DataModuleConfig.fromTrainConfig(train_config),
        torch.arange(100) % model_config.vocabSize,
        validationSequence=torch.arange(40) % model_config.vocabSize,
        testSequence=torch.arange(40) % model_config.vocabSize,
    )
    model = TinyGPTLanguageModel(model_config)
    evaluator = Evaluator(
        model,
        data_module,
        train_config,
        EarlyStopping(patience=2, delta=0.0),
    )
    trainer = LMTrainer(
        model_config,
        train_config,
        model,
        data_module,
        evaluator=evaluator,
    )
    trainer.bestValLoss = 1.0
    trainer.checkpoints.saveCheckpoint(
        trainer.model,
        trainer.optimizer,
        trainer.lrStrategy.state_dict(),
        step=7,
        bestValLoss=1.0,
        path=str(checkpoint),
    )

    test_loss = trainer.evaluateBestCheckpointOnTest()

    assert test_loss is not None
    assert test_loss > 0
    assert '"type": "test"' in (
        run_directory / "metrics.jsonl"
    ).read_text(encoding="utf-8")


def test_best_checkpoint_tracks_lower_loss_below_early_stop_delta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    model_config = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=8,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    train_config = TrainConfig(
        batchSize=2,
        maxSteps=2,
        evalInterval=1,
        evalIters=1,
        earlyStopPatience=2,
        earlyStopDelta=0.003,
        plotCurve=False,
        ckptPath=str(checkpoint),
        validationDataPath=None,
        testDataPath=None,
        device="cpu",
    )
    data_module = SequenceDataModule(
        model_config,
        DataModuleConfig.fromTrainConfig(train_config),
        torch.arange(100) % model_config.vocabSize,
    )
    model = TinyGPTLanguageModel(model_config)
    evaluator = Evaluator(
        model,
        data_module,
        train_config,
        EarlyStopping(patience=2, delta=0.003),
    )
    losses = iter(
        [
            {"train": 1.8, "val": 1.8025},
            {"train": 1.79, "val": 1.7989},
        ]
    )
    monkeypatch.setattr(evaluator, "estimate_loss", lambda: next(losses))
    trainer = LMTrainer(
        model_config,
        train_config,
        model,
        data_module,
        evaluator=evaluator,
    )
    trainer.callbacks = [
        CheckpointCallback(buildCheckpointContext(trainer), logging.getLogger("test"))
    ]
    monkeypatch.setattr(trainer, "_trainStep", lambda: 0.0)

    trainer.train()

    saved = Checkpoint.load(str(checkpoint), "cpu")
    assert saved.step == 1
    assert saved.bestValLoss is not None
    assert abs(saved.bestValLoss - 1.7989) < 1e-9
    assert evaluator.early_stopping.noImproveEvals == 1


def test_exhausted_early_stopping_requires_explicit_reset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    model_config = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=8,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    train_config = TrainConfig(
        batchSize=2,
        maxSteps=10,
        evalInterval=1,
        evalIters=1,
        earlyStopPatience=2,
        plotCurve=False,
        ckptPath=str(checkpoint),
        validationDataPath=None,
        testDataPath=None,
        device="cpu",
    )
    data_module = SequenceDataModule(
        model_config,
        DataModuleConfig.fromTrainConfig(train_config),
        torch.arange(100) % model_config.vocabSize,
    )

    def make_trainer() -> LMTrainer:
        model = TinyGPTLanguageModel(model_config)
        evaluator = Evaluator(
            model,
            data_module,
            train_config,
            EarlyStopping(patience=2, delta=0.003),
        )
        return LMTrainer(
            model_config,
            train_config,
            model,
            data_module,
            evaluator=evaluator,
        )

    initial = make_trainer()
    initial.checkpoints.saveCheckpoint(
        initial.model,
        initial.optimizer,
        initial.lrStrategy.state_dict(),
        step=5,
        bestValLoss=1.0,
        earlyStoppingState={
            "noImproveEvals": 2,
            "referenceLoss": 1.0,
        },
        path=initial.checkpoints.latestPath,
    )

    stopped = make_trainer()
    stopped.loadCheckpointIfExists()
    assert stopped.evaluator is not None and stopped.evaluator.early_stopping.is_exhausted()

    def fail_train() -> float:
        raise AssertionError("completed run should not train")

    monkeypatch.setattr(stopped, "_trainStep", fail_train)
    stopped.train()
    assert stopped.trainingCurve == []

    continued = make_trainer()
    continued.loadCheckpointIfExists(resetEarlyStopping=True)
    assert continued.evaluator is not None
    assert not continued.evaluator.early_stopping.is_exhausted()
    assert continued.evaluator.early_stopping.noImproveEvals == 0
