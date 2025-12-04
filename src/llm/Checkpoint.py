from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any, cast
import logging
from dataclasses import dataclass

import torch

from .Config import ModelConfig, TrainConfig
from .Model import TinyGPTLanguageModel

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
    generatorState: Optional[torch.Tensor] = None

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
            "generatorState": self.generatorState,
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
            generatorState=data.get("generatorState", None),
        )

    def save(self, path: str, device: str | torch.device) -> None:
        torch.save(self.toDict(), path)  # pyright: ignore[reportUnknownMemberType]

    def exportModel(self, out_path: str) -> None:
        torch.save(self.modelState, out_path)  # pyright: ignore[reportUnknownMemberType]

    @staticmethod
    def load(path: str, device: str | torch.device) -> "Checkpoint":
        try:
            data = cast(Dict[str, Any], torch.load(path, map_location=device, weights_only=False))  # pyright: ignore[reportUnknownMemberType]
        except TypeError:
            data = cast(Dict[str, Any], torch.load(path, map_location=device))  # pyright: ignore[reportUnknownMemberType]
        return Checkpoint.fromDict(data)

    @staticmethod
    def fromTrainingState(
        model: TinyGPTLanguageModel,
        optimizer: torch.optim.Optimizer,
        modelConfig: Optional[ModelConfig],
        trainConfig: Optional[TrainConfig],
        step: int,
        bestValLoss: Optional[float],
        lrStrategyState: Optional[Dict[str, Any]] = None,
        generatorState: Optional[torch.Tensor] = None,
        version: int = CHECKPOINT_VERSION,
    ) -> "Checkpoint":
        return Checkpoint(
            version=version,
            modelState=model.state_dict(),
            optimizerState=optimizer.state_dict(),
            step=step,
            bestValLoss=bestValLoss,
            modelConfig=modelConfig.__dict__ if modelConfig is not None else {},
            trainConfig=trainConfig.__dict__ if trainConfig is not None else {},
            lrStrategyState=lrStrategyState,
            generatorState=generatorState,
        )


class CheckpointManager:
    def __init__(
        self,
        modelCfg: ModelConfig,
        trainCfg: TrainConfig,
        trainCkptPath: Optional[str] = None,
        modelCkptPath: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.modelCfg = modelCfg
        self.trainCfg = trainCfg
        self.trainCkptPath = trainCkptPath or trainCfg.ckptPath or "checkpoints/tiny_train.pt"
        self.modelCkptPath = modelCkptPath or "checkpoints/tiny_model.pt"
        self.logger = logger or logging.getLogger(__name__)

        for path in (self.trainCkptPath, self.modelCkptPath):
            ckptDir = os.path.dirname(path)
            if ckptDir:
                os.makedirs(ckptDir, exist_ok=True)

    def saveCheckpoint(
        self,
        model: TinyGPTLanguageModel,
        optimizer: torch.optim.Optimizer,
        lrStrategyState: Optional[Dict[str, Any]],
        step: int,
        bestValLoss: Optional[float],
        generatorState: Optional[torch.Tensor] = None,
    ) -> None:
        checkpoint: Checkpoint = Checkpoint.fromTrainingState(
            model=model,
            optimizer=optimizer,
            modelConfig=self.modelCfg,
            trainConfig=self.trainCfg,
            step=step,
            bestValLoss=bestValLoss,
            lrStrategyState=lrStrategyState,
            generatorState=generatorState,
            version=CHECKPOINT_VERSION,
        )
        checkpoint.save(self.trainCkptPath, self.trainCfg.device)

    def loadCheckpoint(
        self,
        model: TinyGPTLanguageModel,
        optimizer: torch.optim.Optimizer,
        lrStrategy: Optional[Any] = None,
    ) -> Tuple[int, Optional[float], bool, int, bool, Dict[str, Dict[str, Any]], Optional[torch.Tensor]]:
        if not os.path.exists(self.trainCkptPath):
            return 0, None, False, CHECKPOINT_VERSION, True, {}, None

        checkpoint = Checkpoint.load(self.trainCkptPath, self.trainCfg.device)
        model.load_state_dict(checkpoint.modelState)
        optimizer.load_state_dict(checkpoint.optimizerState)
        step = checkpoint.step
        bestValLoss = checkpoint.bestValLoss

        version = checkpoint.version
        version_matches = version == CHECKPOINT_VERSION
        generator_state = checkpoint.generatorState

        lrStateRestored = False
        if lrStrategy is not None and version_matches:
            schedState = checkpoint.lrStrategyState
            if schedState is not None:
                lrStrategy.load_state_dict(schedState)
                lrStateRestored = True

        configDrift: Dict[str, Dict[str, Any]] = {}
        savedModelConfig = checkpoint.modelConfig
        savedTrainConfig = checkpoint.trainConfig
        if savedModelConfig:
            configDrift["model"] = {
                k: v
                for k, v in savedModelConfig.items()
                if k in self.modelCfg.__dict__ and self.modelCfg.__dict__[k] != v
            }
        if savedTrainConfig:
            configDrift["train"] = {
                k: v
                for k, v in savedTrainConfig.items()
                if k in self.trainCfg.__dict__ and self.trainCfg.__dict__[k] != v
            }

        return step, bestValLoss, lrStateRestored, version, version_matches, configDrift, generator_state

    def saveModel(self, out_path: Optional[str] = None) -> None:
        if not os.path.exists(self.trainCkptPath):
            raise FileNotFoundError(self.trainCkptPath)

        checkpoint = Checkpoint.load(self.trainCkptPath, self.trainCfg.device)
        checkpoint.exportModel(out_path or self.modelCkptPath)

    def loadModel(self, model: TinyGPTLanguageModel, modelPath: Optional[str] = None) -> None:
        path = modelPath or self.modelCkptPath
        if not os.path.exists(path):
            if os.path.exists(self.trainCkptPath):
                checkpoint = Checkpoint.load(self.trainCkptPath, self.trainCfg.device)
                checkpoint.exportModel(self.modelCkptPath)
                path = self.modelCkptPath
            else:
                raise FileNotFoundError(
                    f"Model checkpoint not found at {path} and no training checkpoint at {self.trainCkptPath}"
                )

        state = torch.load(  # pyright: ignore[reportUnknownMemberType]
            path,
            map_location=self.trainCfg.device,
            weights_only=True,
        )
        model_state: Dict[str, Any]
        if isinstance(state, dict) and "modelState" in state:
            model_state = cast(Dict[str, Any], state["modelState"])
        else:
            model_state = cast(Dict[str, Any], state)
        model.load_state_dict(model_state)
