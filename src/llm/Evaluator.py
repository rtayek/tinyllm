from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
import logging

import torch

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
        self, model: TinyGPTLanguageModel, data_module: SequenceDataModule, trainConfig: TrainConfig, early_stopping: EarlyStopping, generator: Optional[torch.Generator] = None, logger: Optional[logging.Logger] = None
    ) -> None:
        self.model = model
        self.dataModule = data_module
        self.trainConfig = trainConfig
        self.early_stopping = early_stopping
        self.generator = generator or torch.Generator()
        self.logger = logger or logging.getLogger(__name__)

    def estimate_loss(self) -> Dict[str, float]:
        return {split: self.estimate_split(split) for split in ("train", "val")}

    def estimate_split(
        self,
        split: str,
        generator: torch.Generator | None = None,
    ) -> float:
        was_training = self.model.training
        loss_list: List[float] = []
        activeGenerator = generator if generator is not None else self.generator
        try:
            self.model.eval()
            with torch.no_grad():
                for _ in range(self.trainConfig.evalIters):
                    batchX, batchY = self.dataModule.getBatch(split, activeGenerator)
                    _, loss, _ = self.model(batchX, batchY)
                    if loss is None:
                        raise RuntimeError(f"Loss is None for split '{split}'")
                    loss_list.append(float(loss.item()))
        finally:
            if was_training:
                self.model.train()
        return sum(loss_list) / float(len(loss_list))

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
