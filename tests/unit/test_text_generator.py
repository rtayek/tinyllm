from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from llm.TextGenerator import AutoregressiveGenerator


class RecordingModel:
    def __init__(self) -> None:
        self.indices: Tensor | None = None
        self.max_new_tokens: int | None = None
        self.temperature: float | None = None
        self.top_k: int | None = None
        self.seed: int | None = None

    def generate_autoregressive(
        self,
        indices: Tensor,
        maxNewTokens: int,
        temperature: float = 1.0,
        topK: int | None = None,
        seed: int | None = None,
    ) -> Tensor:
        self.indices = indices.clone()
        self.max_new_tokens = maxNewTokens
        self.temperature = temperature
        self.top_k = topK
        self.seed = seed
        return indices


class FixedOutputModel(RecordingModel):
    def __init__(self, output: bytes) -> None:
        super().__init__()
        self.output = output

    def generate_autoregressive(
        self,
        indices: Tensor,
        maxNewTokens: int,
        temperature: float = 1.0,
        topK: int | None = None,
        seed: int | None = None,
    ) -> Tensor:
        return torch.tensor([list(self.output)], dtype=torch.long)


def test_generate_text_uses_utf8_prompt_bytes() -> None:
    model = RecordingModel()
    generator = AutoregressiveGenerator(model)  # type: ignore[arg-type]

    text = generator.generateText(prompt="Holmés", maxNewTokens=12)

    assert model.indices is not None
    expected = torch.tensor([list("Holmés".encode("utf-8"))], dtype=torch.long)
    assert torch.equal(model.indices, expected)
    assert model.max_new_tokens == 12
    assert text == "Holmés"


def test_generate_text_uses_zero_token_for_empty_prompt() -> None:
    model = RecordingModel()
    generator = AutoregressiveGenerator(model)  # type: ignore[arg-type]

    data = generator.generateBytes(prompt="", maxNewTokens=0)

    assert model.indices is not None
    assert torch.equal(model.indices, torch.zeros((1, 1), dtype=torch.long))
    assert data == b"\x00"


def test_constructor_ignores_legacy_device_argument() -> None:
    model = RecordingModel()
    generator = AutoregressiveGenerator(model, "cpu")  # type: ignore[arg-type]

    assert generator.generateBytes(prompt="", maxNewTokens=0) == b"\x00"


def test_generate_text_forwards_sampling_options() -> None:
    model = RecordingModel()
    generator = AutoregressiveGenerator(model)  # type: ignore[arg-type]

    generator.generateText(
        prompt="Holmes",
        maxNewTokens=5,
        temperature=0.8,
        topK=50,
        seed=123,
    )

    assert model.temperature == 0.8
    assert model.top_k == 50
    assert model.seed == 123


def test_save_sample_creates_tmp_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    model = RecordingModel()
    generator = AutoregressiveGenerator(model)  # type: ignore[arg-type]

    generator.saveSample(prompt="Holmes", maxNewTokens=0)

    assert (tmp_path / "tmp" / "sample.txt").read_text(
        encoding="utf-8"
    ) == "Holmes"


def test_save_sample_accepts_run_specific_path(tmp_path: Path) -> None:
    model = RecordingModel()
    generator = AutoregressiveGenerator(model)  # type: ignore[arg-type]
    path = tmp_path / "runs" / "experiment" / "samples" / "sample.txt"

    generator.saveSample(prompt="Holmes", maxNewTokens=0, path=path)

    assert path.read_text(encoding="utf-8") == "Holmes"


def test_generate_text_replaces_invalid_utf8_by_default() -> None:
    model = FixedOutputModel(b"A\xffB")
    generator = AutoregressiveGenerator(model)  # type: ignore[arg-type]

    assert generator.generateBytes(maxNewTokens=0) == b"A\xffB"
    assert generator.generateText(maxNewTokens=0) == "A\ufffdB"


def test_generate_text_allows_ignore_override() -> None:
    model = FixedOutputModel(b"A\xffB")
    generator = AutoregressiveGenerator(model)  # type: ignore[arg-type]

    assert generator.generateText(maxNewTokens=0, errors="ignore") == "AB"
