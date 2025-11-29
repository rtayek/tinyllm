from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Any

import torch

from Config import TrainConfig
from Model import TinyGpt
from DataModule import ByteDataModule
from EarlyStopping import EarlyStopping


@dataclass
class EvalResult:
    step: int
    train_loss: float
    val_loss: float
    frac_improvement: Optional[float]
    improved: bool
    should_stop: bool
    no_improve_evals: int

    def toDict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "EvalResult":
        return cls(
            step=int(data.get("step", 0)),
            train_loss=float(data.get("train_loss", 0.0)),
            val_loss=float(data.get("val_loss", 0.0)),
            frac_improvement=data.get("frac_improvement", None),
            improved=bool(data.get("improved", False)),
            should_stop=bool(data.get("should_stop", False)),
            no_improve_evals=int(data.get("no_improve_evals", 0)),
        )


class Evaluator:
    def __init__(
        self,
        model: TinyGpt,
        data_module: ByteDataModule,
        train_cfg: TrainConfig,
        early_stopping: EarlyStopping,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.model = model
        self.data_module = data_module
        self.train_cfg = train_cfg
        self.early_stopping = early_stopping
        self.generator = generator or torch.Generator()

    def estimate_loss(self) -> Dict[str, float]:
        self.model.eval()
        losses: Dict[str, float] = {}

        with torch.no_grad():
            for split in ("train", "val"):
                loss_list: List[float] = []
                for _ in range(self.train_cfg.evalIters):
                    batchX, batchY = self.data_module.getBatch(split, self.generator)
                    _, loss = self.model(batchX, batchY)
                    if loss is None:
                        raise RuntimeError("Loss is None in estimateLoss")
                    loss_list.append(float(loss.item()))
                losses[split] = sum(loss_list) / float(len(loss_list))

        self.model.train()
        return losses

    def evaluate(self, step: int, best_val_loss: Optional[float]) -> EvalResult:
        losses = self.estimate_loss()
        trainLoss = losses["train"]
        valueLoss = losses["val"]

        improved, frac_improvement, should_stop, no_improve = self.early_stopping.check(
            best_val_loss, valueLoss
        )

        return EvalResult(
            step=step,
            train_loss=trainLoss,
            val_loss=valueLoss,
            frac_improvement=frac_improvement,
            improved=improved,
            should_stop=should_stop,
            no_improve_evals=no_improve,
        )
