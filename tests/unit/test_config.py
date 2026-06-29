from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from llm.Config import ModelConfig, RunConfig, TrainConfig, RunPaths


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nEmbed": 0}, "nEmbed"),
        ({"nHead": 0}, "nHead"),
        ({"nLayer": 0}, "nLayer"),
        ({"nEmbed": 10, "nHead": 3}, "divisible"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_model_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelConfig(**cast(dict[str, Any], kwargs))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"maxSteps": 0}, "maxSteps"),
        ({"evalInterval": 0}, "evalInterval"),
        ({"snapshotInterval": -1}, "snapshotInterval"),
        ({"maxSnapshots": -1}, "maxSnapshots"),
        ({"weightDecay": -0.1}, "weightDecay"),
        ({"earlyStopPatience": 0}, "earlyStopPatience"),
        ({"earlyStopDelta": -0.1}, "earlyStopDelta"),
    ],
)
def test_train_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TrainConfig(**cast(dict[str, Any], kwargs))


def test_train_config_dict_round_trips() -> None:
    config = TrainConfig(
        seed=42,
        batchSize=4,
        learningRate=1e-3,
        warmupFrac=0.25,
        maxSteps=12,
        evalInterval=3,
        evalIters=5,
        snapshotInterval=6,
        maxSnapshots=2,
        weightDecay=0.03,
        earlyStopPatience=7,
        earlyStopDelta=0.002,
        plotCurve=False,
        dataModule="token",
        runDir="runs/example",
        ckptPath="runs/example/checkpoints/best.pt",
        dataPath="data/train.txt",
        validationDataPath="data/validation.txt",
        testDataPath="data/test.txt",
        device="cpu",
    )

    assert TrainConfig.fromDict(config.toDict()) == config


def test_train_config_run_json_dict_round_trips_with_default_paths() -> None:
    config = TrainConfig(
        seed=42,
        batchSize=4,
        learningRate=1e-3,
        warmupFrac=0.25,
        maxSteps=12,
        evalInterval=3,
        evalIters=5,
        snapshotInterval=6,
        maxSnapshots=2,
        weightDecay=0.03,
        earlyStopPatience=7,
        earlyStopDelta=0.002,
        plotCurve=False,
        dataModule="token",
        runDir="runs/example",
        ckptPath="runs/example/checkpoints/best.pt",
        device="cpu",
    )

    assert TrainConfig.fromDict(config.toRunJsonDict()) == config


def test_train_config_from_dict_ignores_unknown_keys() -> None:
    config = TrainConfig.fromDict(
        {
            "seed": 42,
            "device": "cpu",
            "futureField": "ignored",
            "schema_version": 3,
        }
    )

    assert config.seed == 42
    assert config.device == "cpu"


def test_train_config_from_dict_restores_omitted_defaults() -> None:
    config = TrainConfig.fromDict({"seed": 42, "device": "cpu"})

    assert config.seed == 42
    assert config.device == "cpu"
    assert config.dataPath == TrainConfig().dataPath
    assert config.validationDataPath == TrainConfig().validationDataPath
    assert config.testDataPath == TrainConfig().testDataPath


def test_run_config_loads_replayable_run_json(tmp_path: Path) -> None:
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": {
                    "vocabSize": 256,
                    "blockSize": 32,
                    "nEmbed": 64,
                    "nHead": 4,
                    "nLayer": 2,
                    "dropout": 0.1,
                    "use_cache": False,
                },
                "training": {
                    "seed": 99,
                    "batchSize": 8,
                    "learningRate": 0.001,
                    "warmupFrac": 0.2,
                    "maxSteps": 10,
                    "evalInterval": 2,
                    "evalIters": 3,
                    "snapshotInterval": 5,
                    "maxSnapshots": 2,
                    "weightDecay": 0.01,
                    "earlyStopPatience": 4,
                    "earlyStopDelta": 0.0,
                    "plotCurve": False,
                    "dataModule": "token",
                    "runDir": "runs/example",
                    "ckptPath": "runs/example/checkpoints/best.pt",
                    "device": "cpu",
                },
                "corpora": {
                    "train": {"path": "data/train.txt"},
                    "validation": {"path": "data/val.txt"},
                    "test": {"path": "data/test.txt"},
                },
            }
        ),
        encoding="utf-8",
    )

    run_config = RunConfig.fromRunJson(run_json)

    assert run_config.modelConfig.blockSize == 32
    assert run_config.trainConfig.seed == 99
    assert run_config.trainConfig.runDir == "runs/example"
    assert run_config.trainConfig.dataPath == "data/train.txt"
    assert run_config.trainConfig.validationDataPath == "data/val.txt"
    assert run_config.trainConfig.testDataPath == "data/test.txt"


def test_run_paths_from_train_config_copies_path_fields() -> None:
    config = TrainConfig(
        ckptPath="runs/example/checkpoints/best.pt",
        runDir="runs/example",
        dataPath="data/train.txt",
        validationDataPath="data/validation.txt",
        testDataPath="data/test.txt",
        device="cpu",
    )
    paths = config.paths()
    assert isinstance(paths, RunPaths)
    assert paths.dataPath == config.dataPath
    assert paths.validationDataPath == config.validationDataPath
    assert paths.testDataPath == config.testDataPath
    assert paths.ckptPath == config.ckptPath
    assert paths.runDir == config.runDir


def test_run_paths_handles_none_optional_paths() -> None:
    config = TrainConfig(
        validationDataPath=None,
        testDataPath=None,
        device="cpu",
    )
    paths = config.paths()
    assert paths.validationDataPath is None
    assert paths.testDataPath is None


def test_run_directory_delegates_to_run_paths() -> None:
    # TrainConfig.runDirectory must agree with RunPaths.runDirectory after the
    # delegation refactor (single source of truth).
    config = TrainConfig(
        ckptPath="runs/example/checkpoints/best.pt",
        device="cpu",
    )
    assert config.runDirectory() == config.paths().runDirectory()
    assert config.runDirectory() == Path("runs/example")


def test_run_paths_run_directory_prefers_explicit_run_dir() -> None:
    config = TrainConfig(
        runDir="runs/explicit",
        ckptPath="somewhere/custom-name.pt",
        device="cpu",
    )

    assert config.paths().runDirectory() == Path("runs/explicit")
    assert config.runDirectory() == Path("runs/explicit")


def test_run_paths_run_directory_none_for_nonstandard_layout() -> None:
    config = TrainConfig(ckptPath="best.pt", device="cpu")
    assert config.paths().runDirectory() is None
