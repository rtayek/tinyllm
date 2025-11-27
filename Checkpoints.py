# Checkpoints.py

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any, cast

import torch

from Config import ModelConfig, TrainConfig
from Model import TinyGpt

CHECKPOINT_VERSION = 1


class CheckpointManager:
    def __init__(self, modelCfg: ModelConfig, trainCfg: TrainConfig) -> None:
        self.modelCfg = modelCfg
        self.trainCfg = trainCfg
        ckptDir = os.path.dirname(trainCfg.ckptPath)
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
            "version": CHECKPOINT_VERSION,
            "modelState": model.state_dict(),
            "optimizerState": optimizer.state_dict(),
            "step": step,
            "bestValLoss": bestValLoss,
            "modelConfig": self.modelCfg.__dict__,
            "trainConfig": self.trainCfg.__dict__,
            "lrStrategyState": lrStrategyState,
        }
        torch.save(  # pyright: ignore[reportUnknownMemberType]
            checkpoint,
            self.trainCfg.ckptPath,
        )

    def load(
        self,
        model: TinyGpt,
        optimizer: torch.optim.Optimizer,
        lrStrategy: Optional[Any] = None,
    ) -> Tuple[int, Optional[float], bool, int, bool, Dict[str, Dict[str, Any]]]:
        if not os.path.exists(self.trainCfg.ckptPath):
            return 0, None, False, CHECKPOINT_VERSION, True, {}

        checkpoint = cast(
            Dict[str, Any],
            torch.load(  # pyright: ignore[reportUnknownMemberType]
                self.trainCfg.ckptPath,
                map_location=self.trainCfg.device,
            ),
        )
        model.load_state_dict(checkpoint["modelState"])
        optimizer.load_state_dict(checkpoint["optimizerState"])
        step = int(checkpoint.get("step", 0))
        bestValLoss = checkpoint.get("bestValLoss", None)

        version = int(checkpoint.get("version", CHECKPOINT_VERSION))
        version_matches = version == CHECKPOINT_VERSION

        lr_state_restored = False
        if lrStrategy is not None and version_matches:
            schedState = checkpoint.get("lrStrategyState", None)
            if schedState is not None:
                lrStrategy.load_state_dict(schedState)
                lr_state_restored = True

        config_drift: Dict[str, Dict[str, Any]] = {}
        saved_model_cfg = checkpoint.get("modelConfig", None)
        saved_train_cfg = checkpoint.get("trainConfig", None)
        if saved_model_cfg is not None:
            config_drift["model"] = {
                k: v
                for k, v in saved_model_cfg.items()
                if k in self.modelCfg.__dict__ and self.modelCfg.__dict__[k] != v
            }
        if saved_train_cfg is not None:
            config_drift["train"] = {
                k: v
                for k, v in saved_train_cfg.items()
                if k in self.trainCfg.__dict__ and self.trainCfg.__dict__[k] != v
            }

        return step, bestValLoss, lr_state_restored, version, version_matches, config_drift
