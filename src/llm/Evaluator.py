from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import logging

import torch
import torch.nn.functional as F

from .Config import TrainConfig
from .Model import TinyGPTLanguageModel
from .DataModule import SequenceDataModule
from .EarlyStopping import EarlyStopping, EarlyStopResult
from .EvalResult import EvalResult as LossEvalResult


@dataclass
class TrainEvalResult:
    """Result of a training-loop evaluation step.

    Distinct from ``EvalResult`` in ``llm.EvalResult``, which is the richer
    frozen dataclass used by research scripts and the ``EvaluationMode``
    evaluator hierarchy.
    """
    step: int
    train_loss: float
    val_loss: float
    frac_improvement: Optional[float]
    improved: bool
    should_stop: bool
    no_improve_evals: int



class Evaluator:
    def __init__(
        self,
        model: TinyGPTLanguageModel,
        data_module: SequenceDataModule,
        trainConfig: TrainConfig,
        early_stopping: EarlyStopping,
        generator: Optional[torch.Generator] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.dataModule = data_module
        self.trainConfig = trainConfig
        self.early_stopping = early_stopping
        self.generator = generator or torch.Generator()
        self.logger = logger or logging.getLogger(__name__)

    def _accumulate_losses(self, split: str, generator: torch.Generator) -> float:
        loss_list: list[float] = []
        for _ in range(self.trainConfig.evalIters):
            batchX, batchY = self.dataModule.getBatch(split, generator)
            _, loss, _ = self.model(batchX, batchY)
            if loss is None:
                raise RuntimeError(f"Loss is None for split '{split}'")
            loss_list.append(float(loss.item()))
        return sum(loss_list) / float(len(loss_list))

    def estimate_loss(self) -> Dict[str, float]:
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                return {split: self._accumulate_losses(split, self.generator) for split in ("train", "val")}
        finally:
            if was_training:
                self.model.train()

    def estimate_split(
        self,
        split: str,
        generator: torch.Generator | None = None,
    ) -> float:
        was_training = self.model.training
        activeGenerator = generator if generator is not None else self.generator
        try:
            self.model.eval()
            with torch.no_grad():
                return self._accumulate_losses(split, activeGenerator)
        finally:
            if was_training:
                self.model.train()

    def estimate_split_result(
        self,
        split: str,
        generator: torch.Generator | None = None,
        name: str | None = None,
        checkpoint: str | None = None,
        corpus: str | None = None,
        notes: str | None = None,
    ) -> LossEvalResult:
        loss = self.estimate_split(split, generator)
        return LossEvalResult(
            name=name or split,
            split=split,
            loss=loss,
            nTokens=self.trainConfig.evalIters
            * self.trainConfig.batchSize
            * self.dataModule.modelConfig.blockSize,
            nWindows=self.trainConfig.evalIters * self.trainConfig.batchSize,
            method="sampled",
            checkpoint=checkpoint,
            corpus=corpus,
            notes=notes,
        )

    def estimate_split_full(
        self,
        split: str,
        batch_size: int | None = None,
        stride: int | None = None,
    ) -> float:
        loss, _total_tokens, _n_windows, _method = self._estimate_split_full_stats(
            split,
            batch_size=batch_size,
            stride=stride,
        )
        return loss

    def estimate_split_full_result(
        self,
        split: str,
        batch_size: int | None = None,
        stride: int | None = None,
        name: str | None = None,
        checkpoint: str | None = None,
        corpus: str | None = None,
        notes: str | None = None,
    ) -> LossEvalResult:
        loss, total_tokens, n_windows, method = self._estimate_split_full_stats(
            split,
            batch_size=batch_size,
            stride=stride,
        )
        return LossEvalResult(
            name=name or split,
            split=split,
            loss=loss,
            nTokens=total_tokens,
            nWindows=n_windows,
            method=method,
            checkpoint=checkpoint,
            corpus=corpus,
            notes=notes,
        )

    def _estimate_split_full_stats(
        self,
        split: str,
        batch_size: int | None = None,
        stride: int | None = None,
    ) -> tuple[float, int, int, str]:
        """Deterministic full-pass cross-entropy over a split.

        Windows of ``block_size`` tokens slide across the split with the given
        ``stride``; every target byte is scored exactly once. The result is a
        true held-out average rather than a sampled estimate, and (unlike
        ``estimate_split``/``estimate_loss``) it never advances early-stopping
        state and does not depend on a random generator.

        The ``stride`` controls the context/compute tradeoff:

        - ``stride == block_size`` (the default): non-overlapping windows, the
          cheapest option. Each window scores all ``block_size`` of its
          positions, so the leading positions of every window are predicted
          with little context (the first target sees 1 byte, the second 2, and
          so on). This systematically handicaps the model and slightly
          overstates loss, but it is fast and fair across checkpoints.
        - ``stride < block_size``: overlapping windows. Only the final
          ``stride`` positions of each window are scored (the first window
          scores all of its positions, since nothing precedes it), so every
          scored target after the first window has at least
          ``block_size - stride`` bytes of context. Smaller strides give each
          target more context at proportionally more compute. ``stride == 1``
          is the maximal-context sliding-window perplexity used in the LM
          literature; ``stride == block_size // 2`` is the common compromise.

        Up to ``block_size - 1`` trailing bytes that cannot close a full window
        may go unscored; they are a negligible fraction of any real split.
        """
        source = self.dataModule.splitSequence(split)
        block_size = self.dataModule.modelConfig.blockSize
        if source.size(0) < block_size + 1:
            raise ValueError(
                f"Dataset split '{split}' too small for blockSize {block_size}"
            )

        active_batch_size = batch_size or self.trainConfig.batchSize
        if active_batch_size < 1:
            raise ValueError("batch_size must be greater than zero")

        active_stride = stride if stride is not None else block_size
        if not (1 <= active_stride <= block_size):
            raise ValueError(
                f"stride must be in [1, blockSize={block_size}], got {active_stride}"
            )

        # Window start positions. The last start is the largest index for which
        # a full block_size+1 window (context plus its final target) still fits.
        last_start = source.size(0) - block_size - 1
        starts = list(range(0, last_start + 1, active_stride))

        was_training = self.model.training
        device = self.trainConfig.device
        total_loss = 0.0
        total_tokens = 0
        offsets = torch.arange(block_size)
        # For overlapping windows, score only the last `scored` positions of
        # each window (after the first) so each target is counted once. The
        # number of newly-exposed positions equals the gap between this window's
        # start and the previous one, capped at block_size.
        try:
            self.model.eval()
            with torch.no_grad():
                prev_start: int | None = None
                for batch_start in range(0, len(starts), active_batch_size):
                    batch_starts = starts[batch_start : batch_start + active_batch_size]
                    indices = torch.tensor(batch_starts, dtype=torch.long)
                    positions = indices.unsqueeze(1) + offsets.unsqueeze(0)
                    batch_x = source[positions].to(device)
                    batch_y = source[positions + 1].to(device)
                    logits, _, _ = self.model(batch_x)

                    for row, window_start in enumerate(batch_starts):
                        if prev_start is None:
                            scored = block_size  # first window: score everything
                        else:
                            scored = min(window_start - prev_start, block_size)
                        if scored <= 0:
                            prev_start = window_start
                            continue
                        row_logits = logits[row, -scored:, :]
                        row_targets = batch_y[row, -scored:]
                        loss_sum = F.cross_entropy(
                            row_logits,
                            row_targets,
                            reduction="sum",
                        )
                        total_loss += float(loss_sum.item())
                        total_tokens += int(row_targets.numel())
                        prev_start = window_start
        finally:
            if was_training:
                self.model.train()

        method = "full_nonoverlap" if active_stride == block_size else "full_stride"
        return total_loss / float(total_tokens), total_tokens, len(starts), method

    def evaluate(self, step: int, best_val_loss: Optional[float]) -> TrainEvalResult:
        losses = self.estimate_loss()
        train_loss = losses["train"]
        val_loss = losses["val"]

        stop: EarlyStopResult = self.early_stopping.check(best_val_loss, val_loss)

        return TrainEvalResult(
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            frac_improvement=stop.frac_improvement,
            improved=stop.improved,
            should_stop=stop.should_stop,
            no_improve_evals=stop.no_improve_evals,
        )
