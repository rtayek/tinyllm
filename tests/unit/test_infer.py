import pytest
import torch

from llm.Checkpoint import CheckpointManager
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
) -> None:
    def skip_checkpoint_load(
        self: CheckpointManager,
        model: TinyGPTLanguageModel,
        modelPath: str | None,
    ) -> None:
        return None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        CheckpointManager,
        "loadModel",
        skip_checkpoint_load,
    )
    run_config = RunConfig(
        modelConfig=ModelConfig(
            blockSize=4,
            vocabSize=32,
            nEmbed=8,
            nHead=2,
            nLayer=1,
            dropout=0.0,
        ),
        trainConfig=TrainConfig(device="cuda"),
    )

    generator, resolved_config = build_generator(run_config)

    assert resolved_config.device == "cpu"
    assert generator.device == "cpu"
    assert next(generator.model.parameters()).device.type == "cpu"
