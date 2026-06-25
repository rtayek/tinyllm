from __future__ import annotations

import logging
from pathlib import Path

import pytest

import llm.Main as main_module
from llm.Config import RunConfig, TrainConfig


class FakeTrainer:
    def __init__(self) -> None:
        self.model = object()
        self.trainConfig = TrainConfig(device="cpu")

    def loadCheckpointIfExists(self, resetEarlyStopping: bool = False) -> None:
        self.resetEarlyStopping = resetEarlyStopping

    def train(self) -> None:
        self.trained = True

    def evaluateBestCheckpointOnTest(self) -> None:
        self.evaluated = True


class FakeGenerator:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def saveSample(self, **_kwargs: object) -> None:
        self.saved = True


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
