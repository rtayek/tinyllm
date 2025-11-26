# Checkpoints.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import torch

from Config import ModelConfig
from Model import TinyGpt


class CheckpointManager:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        ckptDir = os.path.dirname(cfg.ckptPath)
        if ckptDir:
            os.makedirs(ckptDir, exist_ok=True)

    def save(
        self,
        model: TinyGpt,
        optimizer: torch.optim.Optimizer,
        step: int,
        bestValLoss: Optional[float],
        lrStrategyState: Optional[Dict[str, Any]] = None,
    ) -> None:
        checkpoint: Dict[str, Any] = {
            "modelState": model.state_dict(),
            "optimizerState": optimizer.state_dict(),
            "step": step,
            "bestValLoss": bestValLoss,
            "config": self.cfg.__dict__,
        }
        if lrStrategyState is not None:
            checkpoint["lrStrategyState"] = lrStrategyState
        torch.save(checkpoint, self.cfg.ckptPath)

    def load(
        self,
        model: TinyGpt,
        optimizer: torch.optim.Optimizer,
        lrStrategy: Optional[Any] = None,
    ) -> Tuple[int, Optional[float], bool]:
        if not os.path.exists(self.cfg.ckptPath):
            return 0, None, False

        checkpoint = torch.load(self.cfg.ckptPath, map_location=self.cfg.device)
        model.load_state_dict(checkpoint["modelState"])
        optimizer.load_state_dict(checkpoint["optimizerState"])
        step = int(checkpoint.get("step", 0))
        bestValLoss = checkpoint.get("bestValLoss", None)

        schedulerRestored = False
        if lrStrategy is not None:
            schedState = checkpoint.get("lrStrategyState", None)
            if schedState is not None:
                lrStrategy.load_state_dict(schedState)
                schedulerRestored = True

        return step, bestValLoss, schedulerRestored
