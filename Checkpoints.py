# Checkpoints.py

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any, cast
from dataclasses import dataclass

import torch

from Config import ModelConfig, TrainConfig
from Model import TinyGpt

CHECKPOINT_VERSION = 1


@dataclass
class Checkpoint:
    version: int
    modelState: Dict[str, Any]
    optimizerState: Dict[str, Any]
    step: int
    bestValLoss: Optional[float]
    modelConfig: Dict[str, Any]
    trainConfig: Dict[str, Any]
    lrStrategyState: Optional[Dict[str, Any]] = None

    def toDict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "modelState": self.modelState,
            "optimizerState": self.optimizerState,
            "step": self.step,
            "bestValLoss": self.bestValLoss,
            "modelConfig": self.modelConfig,
            "trainConfig": self.trainConfig,
            "lrStrategyState": self.lrStrategyState,
        }

    @staticmethod
    def fromDict(data: Dict[str, Any]) -> "Checkpoint":
        return Checkpoint(
            version=int(data.get("version", CHECKPOINT_VERSION)),
            modelState=cast(Dict[str, Any], data["modelState"]),
            optimizerState=cast(Dict[str, Any], data["optimizerState"]),
            step=int(data.get("step", 0)),
            bestValLoss=data.get("bestValLoss", None),
            modelConfig=cast(Dict[str, Any], data.get("modelConfig", {})),
            trainConfig=cast(Dict[str, Any], data.get("trainConfig", {})),
            lrStrategyState=cast(Optional[Dict[str, Any]], data.get("lrStrategyState", None)),
        )

    def save(self, path: str, device: str | torch.device) -> None:
        torch.save(self.toDict(), path)  # pyright: ignore[reportUnknownMemberType]

    @staticmethod
    def load(path: str, device: str | torch.device) -> "Checkpoint":
        data = cast(
            Dict[str, Any],
            torch.load(path, map_location=device),  # pyright: ignore[reportUnknownMemberType]
        )
        return Checkpoint.fromDict(data)

    def exportModel(self, out_path: str) -> None:
        torch.save(self.modelState, out_path)  # pyright: ignore[reportUnknownMemberType]

    @staticmethod
    def fromTrainingState(
        model: TinyGpt,
        optimizer: torch.optim.Optimizer,
        modelConfig: ModelConfig,
        trainConfig: TrainConfig,
        step: int,
        bestValLoss: Optional[float],
        lrStrategyState: Optional[Dict[str, Any]] = None,
        version: int = CHECKPOINT_VERSION,
    ) -> "Checkpoint":
        return Checkpoint(
            version=version,
            modelState=model.state_dict(),
            optimizerState=optimizer.state_dict(),
            step=step,
            bestValLoss=bestValLoss,
            modelConfig=modelConfig.__dict__,
            trainConfig=trainConfig.__dict__,
            lrStrategyState=lrStrategyState,
        )


class CheckpointManager:
    def __init__(
        self,
        modelCfg: ModelConfig,
        trainCfg: TrainConfig,
        trainCkptPath: Optional[str] = None,
        modelCkptPath: Optional[str] = None,
    ) -> None:
        self.modelCfg = modelCfg
        self.trainCfg = trainCfg
        self.trainCkptPath = trainCkptPath or trainCfg.ckptPath or "checkpoints/tiny_train.pt"
        self.modelCkptPath = modelCkptPath or "checkpoints/tiny_model.pt"

        for path in (self.trainCkptPath, self.modelCkptPath):
            ckptDir = os.path.dirname(path)
            if ckptDir:
                os.makedirs(ckptDir, exist_ok=True)

    def saveCheckpoint(
        self,
        model: TinyGpt,
        optimizer: torch.optim.Optimizer,
        lrStrategyState: Optional[Dict[str, Any]],
        step: int,
        bestValLoss: Optional[float],
    ) -> None:
        checkpoint: Checkpoint = Checkpoint.fromTrainingState(
            model=model,
            optimizer=optimizer,
            modelConfig=self.modelCfg,
            trainConfig=self.trainCfg,
            step=step,
            bestValLoss=bestValLoss,
            lrStrategyState=lrStrategyState,
            version=CHECKPOINT_VERSION,
        )
        checkpoint.save(self.trainCkptPath, self.trainCfg.device)

    def loadCheckpoint(
        self,
        model: TinyGpt,
        optimizer: torch.optim.Optimizer,
        lrStrategy: Optional[Any] = None,
    ) -> Tuple[int, Optional[float], bool, int, bool, Dict[str, Dict[str, Any]]]:
        if not os.path.exists(self.trainCkptPath):
            return 0, None, False, CHECKPOINT_VERSION, True, {}

        checkpoint = Checkpoint.load(self.trainCkptPath, self.trainCfg.device)
        model.load_state_dict(checkpoint.modelState)
        optimizer.load_state_dict(checkpoint.optimizerState)
        step = checkpoint.step
        bestValLoss = checkpoint.bestValLoss

        version = checkpoint.version
        version_matches = version == CHECKPOINT_VERSION

        lr_state_restored = False
        if lrStrategy is not None and version_matches:
            schedState = checkpoint.lrStrategyState
            if schedState is not None:
                lrStrategy.load_state_dict(schedState)
                lr_state_restored = True

        config_drift: Dict[str, Dict[str, Any]] = {}
        saved_model_cfg = checkpoint.modelConfig
        saved_train_cfg = checkpoint.trainConfig
        if saved_model_cfg:
            config_drift["model"] = {
                k: v
                for k, v in saved_model_cfg.items()
                if k in self.modelCfg.__dict__ and self.modelCfg.__dict__[k] != v
            }
        if saved_train_cfg:
            config_drift["train"] = {
                k: v
                for k, v in saved_train_cfg.items()
                if k in self.trainCfg.__dict__ and self.trainCfg.__dict__[k] != v
            }

        return step, bestValLoss, lr_state_restored, version, version_matches, config_drift

    def saveModel(self, out_path: Optional[str] = None) -> None:
        """
        Extract model weights from the training checkpoint and save them to a separate file.
        """
        path = out_path or self.modelCkptPath
        if not os.path.exists(self.trainCkptPath):
            raise FileNotFoundError(self.trainCkptPath)

        checkpoint = Checkpoint.load(self.trainCkptPath, self.trainCfg.device)
        checkpoint.exportModel(path)

    def loadModel(self, model: TinyGpt, model_path: str) -> None:
        """
        Load model weights from a model-only checkpoint file.
        Accepts either a pure state_dict or a full checkpoint containing 'modelState'.
        """
        path = model_path or self.modelCkptPath
        if not os.path.exists(path):
            # Fallback to training checkpoint if model-only not present
            if os.path.exists(self.trainCkptPath):
                checkpoint = Checkpoint.load(self.trainCkptPath, self.trainCfg.device)
                checkpoint.exportModel(self.modelCkptPath)
                path = self.modelCkptPath
            else:
                raise FileNotFoundError(path)

        state = torch.load(  # pyright: ignore[reportUnknownMemberType]
            path,
            map_location=self.trainCfg.device,
        )
        model_state: Dict[str, Any]
        if isinstance(state, dict) and "modelState" in state:
            model_state = cast(Dict[str, Any], state["modelState"])
        else:
            model_state = cast(Dict[str, Any], state)
        model.load_state_dict(model_state)
