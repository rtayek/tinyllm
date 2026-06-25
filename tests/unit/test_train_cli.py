from __future__ import annotations

import logging
from pathlib import Path

import pytest

import llm.Main as main_module
from llm.Config import RunConfig, TrainConfig


class FakeRunArtifacts:
    def __init__(self) -> None:
        self.written = False

    def writeRunMetadata(self) -> Path | None:
        self.written = True
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
