import torch
import logging
import pytest
from pathlib import Path
from llm.Checkpoint import Checkpoint, CheckpointManager
from llm.Config import RunConfig, TrainConfig, ModelConfig
from llm.Model import TinyGPTLanguageModel
from llm.tensor_utils import resolve_device
from llm.train_app import buildTrainer


def test_resolve_device_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("cuda", logging.getLogger("test")) == "cpu"


def test_resolve_device_falls_back_for_missing_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert resolve_device("cuda:1", logging.getLogger("test")) == "cpu"


def test_buildTrainer_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("synthetic test corpus\n" * 100, encoding="utf-8")
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        dataPath=str(corpus),
        validationDataPath=None,
        testDataPath=None,
        device="cuda",
    )
    run_cfg = RunConfig(modelConfig=model_cfg, trainConfig=train_cfg)

    trainer = buildTrainer(run_cfg, log=logging.getLogger("test"))

    assert trainer.trainConfig.device == "cpu"


def test_checkpoint_load_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint_path = tmp_path / "ckpt.pt"
    model_config = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=1,
        dropout=0.0,
    )
    train_config = TrainConfig(ckptPath=str(checkpoint_path), device="cuda")
    model = TinyGPTLanguageModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters())

    CheckpointManager(model_config, train_config).saveCheckpoint(
        model,
        optimizer,
        lrStrategyState=None,
        step=1,
        bestValLoss=1.0,
    )

    checkpoint = Checkpoint.load(str(checkpoint_path), "cuda")

    assert checkpoint.step == 1
