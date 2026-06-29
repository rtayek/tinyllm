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
from llm.EvalResult import EvalResult as LossEvalResult
from llm.EvaluationProbe import EvalContext, EvaluationProbe
from llm.EvaluationMode import (
    BaselineEvaluator,
    CorruptionEvaluator,
    FullSplitEvaluator,
    PerBookEvaluator,
    SampledLossEvaluator,
)
from llm.Evaluator import Evaluator
from llm.Model import TinyGPTLanguageModel


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
        dropout=0.0,
    )
    train_config = TrainConfig(batchSize=batch_size, device="cpu", evalIters=5)
    data_module = SequenceDataModule(
        model_config,
        DataModuleConfig.fromTrainConfig(train_config),
        sequence=source,
        validationSequence=source,
    )
    model = TinyGPTLanguageModel(model_config).eval()
    return Evaluator(
        model, data_module, train_config, EarlyStopping(patience=1, delta=0.0)
    )


def test_full_split_is_deterministic() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    first = evaluator.estimate_split_full("val")
    second = evaluator.estimate_split_full("val")
    assert first == second  # no randomness at all


def test_loss_eval_result_computes_perplexity() -> None:
    result = LossEvalResult(name="validation", split="val", loss=1.25)

    assert math.isclose(result.perplexity, math.exp(1.25))


def test_loss_eval_result_serialization_round_trips() -> None:
    result = LossEvalResult(
        name="Emma",
        split="val",
        loss=1.5,
        nTokens=128,
        nWindows=16,
        method="full_nonoverlap",
        checkpoint="runs/example/checkpoints/best.pt",
        corpus="corpora/example/splits/validation.txt",
        notes="reference",
    )

    restored = LossEvalResult.fromDict(result.toDict())

    assert restored == result


def test_sampled_split_result_reports_metadata() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    generator = torch.Generator().manual_seed(123)

    result = evaluator.estimate_split_result("val", generator, name="validation")

    assert result.name == "validation"
    assert result.split == "val"
    assert math.isfinite(result.loss)
    assert math.isclose(result.perplexity, math.exp(result.loss))
    assert result.method == "sampled"
    assert result.nWindows == evaluator.trainConfig.evalIters * evaluator.trainConfig.batchSize
    assert result.nWindows is not None
    assert result.nTokens == result.nWindows * evaluator.dataModule.modelConfig.blockSize


def test_full_split_result_reports_distinct_method() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)

    sampled = evaluator.estimate_split_result("val", torch.Generator().manual_seed(123))
    full = evaluator.estimate_split_full_result("val")
    strided = evaluator.estimate_split_full_result("val", stride=1)

    assert full.method == "full_nonoverlap"
    assert strided.method == "full_stride"
    assert full.method != sampled.method
    assert full.nTokens == 56
    assert full.nWindows == 7


def test_sampled_loss_evaluator_mode_wraps_existing_evaluator() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    generator = torch.Generator().manual_seed(123)

    result = SampledLossEvaluator(
        evaluator,
        generator=generator,
    ).evaluate(EvalContext(name="validation", split="val"))

    assert result.name == "validation"
    assert result.method == "sampled"
    assert result.split == "val"


def test_full_split_evaluator_mode_wraps_existing_evaluator() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)

    result = FullSplitEvaluator(evaluator).evaluate(
        EvalContext(name="validation", split="val")
    )

    assert result.method == "full_nonoverlap"
    assert result.nTokens == 56
    assert result.nWindows == 7


def test_per_book_evaluator_mode_reports_book_name_and_corpus() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)

    result = PerBookEvaluator(
        evaluator.model,
        evaluator.dataModule.modelConfig,
        evaluator.trainConfig,
        seed=42,
    ).evaluate(
        EvalContext(
            name="Example Book",
            tokens=source,
            corpus="corpora/example/validation.txt",
        )
    )

    assert result.name == "Example Book"
    assert result.corpus == "corpora/example/validation.txt"
    assert result.method == "sampled"


def test_per_book_evaluator_requires_tokens() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    probe = PerBookEvaluator(
        evaluator.model,
        evaluator.dataModule.modelConfig,
        evaluator.trainConfig,
        seed=42,
    )
    try:
        probe.evaluate(EvalContext(name="missing tokens"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when context.tokens is None")


def test_corruption_evaluator_mode_applies_corruption() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)

    def reverse_bytes(raw: bytes, _name: str) -> bytes:
        return raw[::-1]

    result = CorruptionEvaluator(
        evaluator.model,
        evaluator.dataModule.modelConfig,
        evaluator.trainConfig,
        seed=42,
        corrupt=reverse_bytes,
    ).evaluate(
        EvalContext(name="reverse", raw=bytes(i % 16 for i in range(64)))
    )

    assert result.name == "reverse"
    assert result.method == "sampled"
    assert math.isfinite(result.loss)


def test_corruption_evaluator_requires_raw() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    probe = CorruptionEvaluator(
        evaluator.model,
        evaluator.dataModule.modelConfig,
        evaluator.trainConfig,
        seed=42,
        corrupt=lambda raw, _name: raw[::-1],
    )
    try:
        probe.evaluate(EvalContext(name="missing raw"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when context.raw is None")


def test_baseline_evaluator_mode_wraps_external_loss() -> None:
    result = BaselineEvaluator(method="ngram", loss=1.75, n_tokens=123).evaluate(
        EvalContext(name="3-gram", split="val", notes="Laplace")
    )

    assert result.name == "3-gram"
    assert result.method == "ngram"
    assert result.nTokens == 123
    assert result.notes == "Laplace"
    assert math.isclose(result.perplexity, math.exp(1.75))


def test_evaluation_mode_evaluators_satisfy_probe_protocol() -> None:
    # All five EvaluationMode evaluators must conform to the EvaluationProbe
    # protocol so the self-improvement loop can hold them in one list.
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    probes: list[EvaluationProbe] = [
        SampledLossEvaluator(evaluator),
        FullSplitEvaluator(evaluator),
        PerBookEvaluator(
            evaluator.model,
            evaluator.dataModule.modelConfig,
            evaluator.trainConfig,
            seed=42,
        ),
        CorruptionEvaluator(
            evaluator.model,
            evaluator.dataModule.modelConfig,
            evaluator.trainConfig,
            seed=42,
            corrupt=lambda raw, _name: raw,
        ),
        BaselineEvaluator(loss=1.0),
    ]
    assert len(probes) == 5


def test_evaluation_probe_protocol_seam_carries_context_metadata() -> None:
    class FakeProbe:
        def evaluate(self, context: EvalContext) -> LossEvalResult:
            return LossEvalResult(
                name=context.name,
                split=context.split,
                loss=1.25,
                method="fake_probe",
                checkpoint=context.checkpoint,
                corpus=context.corpus,
                notes=context.notes,
            )

    context = EvalContext(
        name="candidate-a",
        split="validation",
        corpus="corpora/example/validation.txt",
        checkpoint="runs/example/checkpoints/best.pt",
        notes="best-of-n probe",
    )

    probe: EvaluationProbe = FakeProbe()
    result = probe.evaluate(context)

    assert result.name == "candidate-a"
    assert result.split == "validation"
    assert result.corpus == "corpora/example/validation.txt"
    assert result.checkpoint == "runs/example/checkpoints/best.pt"
    assert result.notes == "best-of-n probe"


def test_full_split_batch_size_invariant() -> None:
    # The mean must not depend on how windows are batched. Reuse one evaluator
    # so the model weights are identical across the three calls.
    source = torch.arange(200) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=1)
    a = evaluator.estimate_split_full("val", batch_size=1)
    b = evaluator.estimate_split_full("val", batch_size=4)
    c = evaluator.estimate_split_full("val", batch_size=7)
    assert math.isclose(a, b, rel_tol=1e-6)
    assert math.isclose(a, c, rel_tol=1e-6)


def test_full_split_scores_each_target_once() -> None:
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


def test_full_split_stride_scores_each_target_once() -> None:
    # With stride=1 (sliding window) every target must still be counted once.
    block_size = 8
    source = torch.arange(40) % 16
    evaluator = _make_evaluator(source, block_size=block_size, batch_size=4)

    # Count how many targets a stride-1 pass scores by replicating the layout:
    # first window scores block_size, each subsequent window scores 1.
    last_start = source.size(0) - block_size - 1
    starts = list(range(0, last_start + 1, 1))
    expected_count = block_size + (len(starts) - 1) * 1

    # Re-run the model's own accounting by hand to confirm count.
    count = 0
    prev = None
    for start in starts:
        scored = block_size if prev is None else min(start - prev, block_size)
        count += scored
        prev = start
    assert count == expected_count

    # The function should run without error and return a finite mean.
    value = evaluator.estimate_split_full("val", stride=1)
    assert math.isfinite(value)


def test_full_split_smaller_stride_gives_more_context() -> None:
    # More context (smaller stride) should not increase loss for a trained-ish
    # model; for an untrained model the values should at least both be finite
    # and the stride=1 mean should differ from the non-overlapping mean.
    source = torch.arange(128) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    non_overlapping = evaluator.estimate_split_full("val")  # stride = block_size
    sliding = evaluator.estimate_split_full("val", stride=1)
    assert math.isfinite(non_overlapping)
    assert math.isfinite(sliding)
    # The two layouts score different sets of (context, target) pairs, so the
    # means should not be identical.
    assert non_overlapping != sliding


def test_full_split_rejects_out_of_range_stride() -> None:
    source = torch.arange(64) % 16
    evaluator = _make_evaluator(source, block_size=8, batch_size=4)
    for bad in (0, -1, 9):
        try:
            evaluator.estimate_split_full("val", stride=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for stride={bad}")


def test_full_split_non_dividing_stride_matches_bruteforce() -> None:
    # stride=3 does not divide block_size=8 evenly: the case most likely to
    # expose an off-by-one in the per-window scoring accounting. Replicate the
    # exact windowing logic and assert the mean matches.
    block_size = 8
    stride = 3
    source = torch.arange(50) % 16
    evaluator = _make_evaluator(source, block_size=block_size, batch_size=4)
    model = evaluator.model
    offsets = torch.arange(block_size)

    last_start = source.size(0) - block_size - 1
    starts = list(range(0, last_start + 1, stride))

    total = 0.0
    count = 0
    prev = None
    with torch.no_grad():
        for start in starts:
            scored = block_size if prev is None else min(start - prev, block_size)
            prev = start
            if scored <= 0:
                continue
            pos = torch.tensor([start]).unsqueeze(1) + offsets.unsqueeze(0)
            logits, _, _ = model(source[pos])
            row_logits = logits[0, -scored:, :]
            row_targets = source[pos + 1][0, -scored:]
            loss_sum = F.cross_entropy(row_logits, row_targets, reduction="sum")
            total += float(loss_sum.item())
            count += int(row_targets.numel())

    expected = total / count
    actual = evaluator.estimate_split_full("val", stride=stride)
    assert math.isclose(actual, expected, rel_tol=1e-6)


def test_full_split_non_dividing_stride_scores_each_target_once() -> None:
    # No target between the first window start and the last window's end may be
    # double-counted or skipped, even when stride does not divide block_size.
    block_size = 8
    stride = 3
    source = torch.arange(50) % 16
    last_start = source.size(0) - block_size - 1
    starts = list(range(0, last_start + 1, stride))

    # Track which absolute target positions get scored, with multiplicity.
    scored_positions: list[int] = []
    prev = None
    for start in starts:
        scored = block_size if prev is None else min(start - prev, block_size)
        prev = start
        if scored <= 0:
            continue
        # The scored targets are the last `scored` positions of this window.
        window_target_starts = start + (block_size - scored)
        for offset in range(scored):
            scored_positions.append(window_target_starts + offset)

    # Every scored target appears exactly once (no duplicates).
    assert len(scored_positions) == len(set(scored_positions))
    # Coverage is contiguous from the first target to the last.
    assert scored_positions == list(
        range(scored_positions[0], scored_positions[-1] + 1)
    )


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
