import torch
import logging
import pytest
from pathlib import Path
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
