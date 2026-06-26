from __future__ import annotations

import logging
from pathlib import Path

import pytest

import llm.train_app as main_module
from llm.Config import ModelConfig, RunConfig, TrainConfig
from llm.DataModule import ByteDataModule, TokenDataModule
from llm.train_cli import parseTrainCli


class FakeRunArtifacts:
    def __init__(self) -> None:
        self.written = False

    def writeRunMetadata(self) -> Path | None:
        self.written = True
        return None

    def runMetadataExists(self) -> bool:
        return False

    def appendContinuation(self) -> Path | None:
        return None


class FakeTrainer:
    def __init__(self) -> None:
        self.model = object()
        self.trainConfig = TrainConfig(device="cpu")
        self.runArtifacts = FakeRunArtifacts()
        self.trainResult: bool | None = True

    def loadCheckpointIfExists(self, resetEarlyStopping: bool = False) -> None:
        self.resetEarlyStopping = resetEarlyStopping

    def train(self) -> bool | None:
        self.trained = True
        return self.trainResult

    def evaluateBestCheckpointOnTest(self) -> None:
        self.evaluated = True


class FakeGenerator:
    save_calls = 0

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def saveSample(self, **_kwargs: object) -> None:
        del _kwargs
        type(self).save_calls += 1


def test_parse_train_cli_builds_run_config(tmp_path: Path) -> None:
    cli_config = parseTrainCli(
        [
            "--corpus",
            "train.txt",
            "--validation-corpus",
            "val.txt",
            "--test-corpus",
            "test.txt",
            "--run-dir",
            str(tmp_path / "runs" / "exp"),
            "--seed",
            "123",
            "--block-size",
            "256",
            "--n-embed",
            "128",
            "--n-head",
            "4",
            "--n-layer",
            "3",
            "--max-steps",
            "10",
            "--early-stop-patience",
            "5",
            "--reset-early-stopping",
            "--snapshot-interval",
            "2",
            "--max-snapshots",
            "1",
            "--plot",
            "--log-level",
            "DEBUG",
        ]
    )

    assert cli_config.logLevel == logging.DEBUG
    assert cli_config.resetEarlyStopping is True
    assert cli_config.runConfig.modelConfig.blockSize == 256
    assert cli_config.runConfig.modelConfig.nEmbed == 128
    assert cli_config.runConfig.modelConfig.nHead == 4
    assert cli_config.runConfig.modelConfig.nLayer == 3
    train_config = cli_config.runConfig.trainConfig
    assert train_config.dataPath == "train.txt"
    assert train_config.validationDataPath == "val.txt"
    assert train_config.testDataPath == "test.txt"
    assert train_config.seed == 123
    assert train_config.maxSteps == 10
    assert train_config.earlyStopPatience == 5
    assert train_config.snapshotInterval == 2
    assert train_config.maxSnapshots == 1
    assert train_config.plotCurve is True
    assert Path(train_config.ckptPath).as_posix().endswith("exp/checkpoints/best.pt")


def test_parse_train_cli_rejects_invalid_shape_flag() -> None:
    with pytest.raises(SystemExit):
        parseTrainCli(["--n-head", "0"])


def test_parse_train_cli_rejects_invalid_log_level() -> None:
    with pytest.raises(SystemExit):
        parseTrainCli(["--log-level", "DEBIG"])


def test_main_accepts_model_shape_and_run_dir_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[RunConfig] = []
    fake_trainer = FakeTrainer()

    def fake_build_trainer(
        runConfig: RunConfig | None = None,
        log: logging.Logger | None = None,
    ) -> FakeTrainer:
        del log
        assert runConfig is not None
        seen.append(runConfig)
        fake_trainer.trainConfig = runConfig.trainConfig
        return fake_trainer

    monkeypatch.setattr(main_module, "buildTrainer", fake_build_trainer)
    FakeGenerator.save_calls = 0
    monkeypatch.setattr(main_module, "AutoregressiveGenerator", FakeGenerator)

    main_module.main(
        [
            "--run-dir",
            str(tmp_path / "runs" / "block256"),
            "--block-size",
            "256",
            "--n-embed",
            "128",
            "--n-head",
            "4",
            "--n-layer",
            "3",
        ]
    )

    assert seen
    run_config = seen[0]
    assert run_config.modelConfig.blockSize == 256
    assert run_config.modelConfig.nEmbed == 128
    assert run_config.modelConfig.nHead == 4
    assert run_config.modelConfig.nLayer == 3
    assert Path(run_config.trainConfig.ckptPath).as_posix().endswith(
        "block256/checkpoints/best.pt"
    )
    assert fake_trainer.runArtifacts.written is True
    assert fake_trainer.evaluated is True
    assert FakeGenerator.save_calls == 1


def test_main_writes_run_metadata_after_checkpoint_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_trainer = FakeTrainer()

    def fake_build_trainer(
        runConfig: RunConfig | None = None,
        log: logging.Logger | None = None,
    ) -> FakeTrainer:
        del runConfig, log
        return fake_trainer

    def fail_load(resetEarlyStopping: bool = False) -> None:
        del resetEarlyStopping
        raise ValueError("bad checkpoint")

    fake_trainer.loadCheckpointIfExists = fail_load
    monkeypatch.setattr(main_module, "buildTrainer", fake_build_trainer)

    with pytest.raises(ValueError, match="bad checkpoint"):
        main_module.main([])

    assert fake_trainer.runArtifacts.written is False


def test_main_skips_final_artifacts_when_training_did_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_trainer = FakeTrainer()
    fake_trainer.trainResult = False

    def fake_build_trainer(
        runConfig: RunConfig | None = None,
        log: logging.Logger | None = None,
    ) -> FakeTrainer:
        del runConfig, log
        return fake_trainer

    monkeypatch.setattr(main_module, "buildTrainer", fake_build_trainer)
    FakeGenerator.save_calls = 0
    monkeypatch.setattr(main_module, "AutoregressiveGenerator", FakeGenerator)

    main_module.main([])

    assert fake_trainer.runArtifacts.written is True
    assert not hasattr(fake_trainer, "evaluated")
    assert FakeGenerator.save_calls == 0


def test_build_data_module_uses_registered_factories(tmp_path: Path) -> None:
    corpus = tmp_path / "input.txt"
    corpus.write_text("hello tiny llm", encoding="utf-8")
    logger = logging.getLogger("test.data-module-factory")
    model_config = ModelConfig(blockSize=4)
    base_config = TrainConfig(
        dataPath=str(corpus),
        validationDataPath=None,
        testDataPath=None,
        device="cpu",
    )

    token_module = main_module.build_data_module(
        model_config,
        base_config,
        logger,
    )
    byte_module = main_module.build_data_module(
        model_config,
        TrainConfig(
            dataModule="byte",
            dataPath=str(corpus),
            validationDataPath=None,
            testDataPath=None,
            device="cpu",
        ),
        logger,
    )

    assert isinstance(token_module, TokenDataModule)
    assert isinstance(byte_module, ByteDataModule)

    with pytest.raises(ValueError, match="Unknown dataModule"):
        main_module.build_data_module(
            model_config,
            TrainConfig(
                dataModule="missing",
                dataPath=str(corpus),
                validationDataPath=None,
                testDataPath=None,
                device="cpu",
            ),
            logger,
        )
