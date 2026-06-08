from __future__ import annotations

import torch
from torch import Tensor

from llm.TextGenerator import AutoregressiveGenerator


class RecordingModel:
    def __init__(self) -> None:
        self.indices: Tensor | None = None
        self.max_new_tokens: int | None = None

    def generate_autoregressive(self, indices: Tensor, maxNewTokens: int) -> Tensor:
        self.indices = indices.clone()
        self.max_new_tokens = maxNewTokens
        return indices


def test_generate_text_uses_utf8_prompt_bytes() -> None:
    model = RecordingModel()
    generator = AutoregressiveGenerator(model, "cpu")  # type: ignore[arg-type]

    text = generator.generateText(prompt="Holmés", maxNewTokens=12)

    assert model.indices is not None
    expected = torch.tensor([list("Holmés".encode("utf-8"))], dtype=torch.long)
    assert torch.equal(model.indices, expected)
    assert model.max_new_tokens == 12
    assert text == "Holmés"


def test_generate_text_uses_zero_token_for_empty_prompt() -> None:
    model = RecordingModel()
    generator = AutoregressiveGenerator(model, "cpu")  # type: ignore[arg-type]

    data = generator.generateBytes(prompt="", maxNewTokens=0)

    assert model.indices is not None
    assert torch.equal(model.indices, torch.zeros((1, 1), dtype=torch.long))
    assert data == b"\x00"
