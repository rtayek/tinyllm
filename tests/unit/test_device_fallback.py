import torch
import logging
import pytest
from llm.Config import RunConfig, TrainConfig, ModelConfig
from llm.Main import buildTrainer
from llm.tensor_utils import resolve_device


def test_resolve_device_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("cuda", logging.getLogger("test")) == "cpu"


def test_buildTrainer_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(device="cuda")  # intentionally request CUDA
    run_cfg = RunConfig(modelConfig=model_cfg, trainConfig=train_cfg)

    trainer = buildTrainer(run_cfg, log=logging.getLogger("test"))

    assert trainer.trainConfig.device == "cpu"
