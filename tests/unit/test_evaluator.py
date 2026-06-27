"""Tests for Evaluator, focused on the deterministic full-split evaluation.

estimate_split_full is used to produce held-out cross-entropy numbers that go
into research reports, so its semantics are pinned down here: every target is
scored exactly once, with non-overlapping windows, deterministically.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import DataModuleConfig, SequenceDataModule
from llm.EarlyStopping import EarlyStopping
from llm.Evaluator import Evaluator


class CountingModel(torch.nn.Module):
    """Minimal model with a real embedding-based logit map.

    Returns (logits, loss, None) to match the TinyGPT calling convention.
    Deterministic given its initialization seed.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.forward_calls = 0
        torch.manual_seed(0)
        self.table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        self.forward_calls += 1
        logits = self.table(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss, None


def _make_evaluator(
    source: torch.Tensor,
    block_size: int,
    batch_size: int,
    vocab_size: int = 16,
) -> Evaluator:
    model_config = ModelConfig(
        vocabSize=vocab_size,
        blockSize=block_size,
        nEmbed=16,
        nHead=2,
        nLayer=1,
    )
    train_config = TrainConfig(batchSize=batch_size, device="cpu", evalIters=5)
    data_module = SequenceDataModule(
        model_config,
        DataModuleConfig.fromTrainConfig(train_config),
        sequence=source,
        validationSequence=source,
    )
    model = CountingModel(vocab_size)
    model.eval()
    return Evaluator(
        model, data_module, train_config, EarlyStopping(patience=1, delta=0.0)
    )


def test_full_split_is_deterministic() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    first = evaluator.estimate_split_full("val")
    second = evaluator.estimate_split_full("val")
    assert first == second  # no randomness at all


def test_full_split_batch_size_invariant() -> None:
    # The mean must not depend on how windows are batched.
    source = torch.arange(200) % 16
    a = _make_evaluator(source, block_size=8, batch_size=1).estimate_split_full("val")
    b = _make_evaluator(source, block_size=8, batch_size=4).estimate_split_full("val")
    c = _make_evaluator(source, block_size=8, batch_size=7).estimate_split_full("val")
    assert math.isclose(a, b, rel_tol=1e-6)
    assert math.isclose(a, c, rel_tol=1e-6)


def test_full_split_scores_each_target_once() -> None:
    # With non-overlapping windows of stride block_size, the number of scored
    # targets is floor over the available windows times block_size.
    block_size = 8
    source = torch.arange(64) % 16  # 64 tokens
    evaluator = _make_evaluator(source, block_size=block_size, batch_size=4)

    # Hand-compute the expected mean using the same non-overlapping layout.
    last_start = source.size(0) - block_size - 1  # 64 - 8 - 1 = 55
    starts = list(range(0, last_start + 1, block_size))  # 0,8,16,24,32,40,48
    offsets = torch.arange(block_size)
    model = evaluator.model

    total = 0.0
    count = 0
    with torch.no_grad():
        for start in starts:
            pos = torch.tensor([start]).unsqueeze(1) + offsets.unsqueeze(0)
            x = source[pos]
            y = source[pos + 1]
            logits, _, _ = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            total += float(loss_sum.item())
            count += int(y.numel())

    expected = total / count
    actual = evaluator.estimate_split_full("val")
    assert math.isclose(actual, expected, rel_tol=1e-6)
    # 7 windows x 8 positions = 56 scored targets
    assert count == 56


def test_full_split_does_not_touch_early_stopping() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    before = evaluator.early_stopping.state_dict()
    evaluator.estimate_split_full("val")
    after = evaluator.early_stopping.state_dict()
    assert before == after


def test_full_split_raises_when_split_too_small() -> None:
    source = torch.arange(8) % 16  # exactly block_size, no room for a target
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    try:
        evaluator.estimate_split_full("val")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for too-small split")


def test_full_split_restores_training_mode() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    evaluator.model.train()
    evaluator.estimate_split_full("val")
    assert evaluator.model.training is True
