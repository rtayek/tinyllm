import torch
from pathlib import Path
from typing import Any

import pytest

from llm.Config import ModelConfig, TrainConfig
from llm.Model import TinyGPTLanguageModel
from llm.Checkpoint import Checkpoint, CheckpointManager


def test_checkpoint_export_and_load(tmp_path: Path) -> None:
    dataPath = tmp_path / "input.txt"
    dataPath.write_bytes(b"hello tiny llm\n" * 20)

    trainCkpt = tmp_path / "train_ckpt.pt"
    modelExport = tmp_path / "model_export.pt"

    modelConfig = ModelConfig(blockSize=8, vocabSize=256, nEmbed=16, nHead=2, nLayer=1, dropout=0.0)
    trainConfig = TrainConfig(
        batchSize=2,
        learningRate=1e-3,
        warmupFrac=0.1,
        maxSteps=2,
        evalInterval=1,
        evalIters=1,
        weightDecay=0.0,
        plotCurve=False,
        ckptPath=str(trainCkpt),
        dataPath=str(dataPath),
        device="cpu",
    )

    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    model = TinyGPTLanguageModel(modelConfig)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainConfig.learningRate, weight_decay=trainConfig.weightDecay)
    manager = CheckpointManager(modelConfig, trainConfig)
    manager.saveCheckpoint(model, optimizer, lrStrategyState=None, step=1, bestValLoss=0.5, generatorState=None)

    checkpoint = Checkpoint.load(str(trainCkpt), trainConfig.device)
    checkpoint.exportModel(str(modelExport))
    assert modelExport.exists()

    loadedModel = TinyGPTLanguageModel(modelConfig)
    modelState = torch.load(  # pyright: ignore[reportUnknownMemberType]
        modelExport,
        map_location=trainConfig.device,
        weights_only=True,
    )
    loadedModel.load_state_dict(modelState)

    for pSaved, pLoaded in zip(model.parameters(), loadedModel.parameters()):
        assert torch.equal(pSaved, pLoaded)


def test_checkpoint_save_preserves_existing_file_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"existing checkpoint")
    checkpoint = Checkpoint(
        version=1,
        modelState={},
        optimizerState={},
        step=0,
        bestValLoss=None,
        modelConfig={},
        trainConfig={},
    )

    def fail_save(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated save failure")

    monkeypatch.setattr(torch, "save", fail_save)

    with pytest.raises(OSError, match="simulated save failure"):
        checkpoint.save(str(checkpoint_path))

    assert checkpoint_path.read_bytes() == b"existing checkpoint"
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []


def test_checkpoint_manager_resumes_latest_checkpoint(
    tmp_path: Path,
) -> None:
    best_path = tmp_path / "best.pt"
    model_config = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=8,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    train_config = TrainConfig(
        ckptPath=str(best_path),
        device="cpu",
    )
    model = TinyGPTLanguageModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(model_config, train_config)

    manager.saveCheckpoint(
        model,
        optimizer,
        lrStrategyState=None,
        step=10,
        bestValLoss=1.0,
        path=manager.ckptPath,
    )
    manager.saveCheckpoint(
        model,
        optimizer,
        lrStrategyState=None,
        step=20,
        bestValLoss=1.0,
        path=manager.latestPath,
    )

    loaded_model = TinyGPTLanguageModel(model_config)
    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters())
    step, *_ = manager.loadCheckpoint(loaded_model, loaded_optimizer)

    assert manager.resumePath() == manager.latestPath
    assert step == 20


def test_checkpoint_normalizes_rng_states_for_cpu_generator() -> None:
    checkpoint = Checkpoint.fromDict(
        {
            "modelState": {},
            "optimizerState": {},
            "generatorState": torch.arange(8, dtype=torch.int64),
            "evaluatorGeneratorState": torch.arange(8, dtype=torch.int16),
        }
    )

    assert checkpoint.generatorState is not None
    assert checkpoint.evaluatorGeneratorState is not None
    assert checkpoint.generatorState.device.type == "cpu"
    assert checkpoint.evaluatorGeneratorState.device.type == "cpu"
    assert checkpoint.generatorState.dtype == torch.uint8
    assert checkpoint.evaluatorGeneratorState.dtype == torch.uint8
