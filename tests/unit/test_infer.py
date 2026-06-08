from pathlib import Path

import pytest
import torch

from llm.Checkpoint import Checkpoint
from llm.Config import ModelConfig, RunConfig, TrainConfig
from llm.infer import build_generator, parse_args
from llm.Model import TinyGPTLanguageModel


def test_parse_args_accepts_prompt_and_token_count() -> None:
    args = parse_args(["--prompt", "Mr. Sherlock Holmes", "--tokens", "25"])

    assert args.prompt == "Mr. Sherlock Holmes"
    assert args.tokens == 25


def test_parse_args_preserves_existing_defaults() -> None:
    args = parse_args([])

    assert args.prompt == ""
    assert args.tokens == 400


def test_parse_args_rejects_negative_token_count() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--tokens", "-1"])


def test_build_generator_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model_config = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=8,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    model = TinyGPTLanguageModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = Checkpoint.fromTrainingState(
        model=model,
        optimizer=optimizer,
        modelConfig=model_config,
        trainConfig=None,
        step=0,
        bestValLoss=None,
    )
    checkpoint.save(str(checkpoint_path), "cpu")
    run_config = RunConfig(
        modelConfig=model_config,
        trainConfig=TrainConfig(
            device="cuda",
            ckptPath=str(checkpoint_path),
        ),
    )

    generator, resolved_config = build_generator(run_config)

    assert resolved_config.device == "cpu"
    assert generator.device == "cpu"
    assert next(generator.model.parameters()).device.type == "cpu"


def test_build_generator_uses_checkpoint_model_config(
    tmp_path: Path,
) -> None:
    checkpoint_config = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=8,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    model = TinyGPTLanguageModel(checkpoint_config)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = Checkpoint.fromTrainingState(
        model=model,
        optimizer=optimizer,
        modelConfig=checkpoint_config,
        trainConfig=None,
        step=1,
        bestValLoss=1.0,
    )
    checkpoint.save(str(checkpoint_path), "cpu")
    run_config = RunConfig(
        modelConfig=ModelConfig(),
        trainConfig=TrainConfig(
            device="cpu",
            ckptPath=str(checkpoint_path),
        ),
    )

    generator, _ = build_generator(run_config)

    assert generator.model.cfg == checkpoint_config
