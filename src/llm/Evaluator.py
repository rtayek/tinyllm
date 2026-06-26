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


@dataclass
class EvalResult:
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

    def estimate_split_full(
        self,
        split: str,
        batch_size: int | None = None,
    ) -> float:
        """Evaluate every valid fixed-width window for a split."""
        source = self.dataModule.splitSequence(split)
        block_size = self.dataModule.modelConfig.blockSize
        high = source.size(0) - block_size
        if high <= 0:
            raise ValueError(
                f"Dataset split '{split}' too small for blockSize {block_size}"
            )

        active_batch_size = batch_size or self.trainConfig.batchSize
        if active_batch_size < 1:
            raise ValueError("batch_size must be greater than zero")

        was_training = self.model.training
        device = self.trainConfig.device
        total_loss = 0.0
        total_tokens = 0
        offsets = torch.arange(block_size)
        try:
            self.model.eval()
            with torch.no_grad():
                for start in range(0, high, active_batch_size):
                    indices = torch.arange(start, min(start + active_batch_size, high))
                    positions = indices.unsqueeze(1) + offsets.unsqueeze(0)
                    batch_x = source[positions].to(device)
                    batch_y = source[positions + 1].to(device)
                    logits, _, _ = self.model(batch_x)
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        batch_y.reshape(-1),
                        reduction="sum",
                    )
                    total_loss += float(loss_sum.item())
                    total_tokens += int(batch_y.numel())
        finally:
            if was_training:
                self.model.train()

        return total_loss / float(total_tokens)

    def evaluate(self, step: int, best_val_loss: Optional[float]) -> EvalResult:
        losses = self.estimate_loss()
        train_loss = losses["train"]
        val_loss = losses["val"]

        stop: EarlyStopResult = self.early_stopping.check(best_val_loss, val_loss)

        return EvalResult(
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            frac_improvement=stop.frac_improvement,
            improved=stop.improved,
            should_stop=stop.should_stop,
            no_improve_evals=stop.no_improve_evals,
        )
