from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, cast
import logging
from dataclasses import dataclass

import torch

from .Config import ModelConfig, TrainConfig
from .Model import TinyGPTLanguageModel

CHECKPOINT_VERSION = 1


def _cpu_rng_state(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError("Checkpoint RNG state must be a tensor")
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


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
    evaluatorGeneratorState: Optional[torch.Tensor] = None
    earlyStoppingState: Optional[Dict[str, Any]] = None

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
            "evaluatorGeneratorState": self.evaluatorGeneratorState,
            "earlyStoppingState": self.earlyStoppingState,
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
            generatorState=_cpu_rng_state(data.get("generatorState", None)),
            evaluatorGeneratorState=_cpu_rng_state(
                data.get("evaluatorGeneratorState", None)
            ),
            earlyStoppingState=cast(
                Optional[Dict[str, Any]],
                data.get("earlyStoppingState", None),
            ),
        )

    def save(self, path: str) -> None:
        target_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(target_dir, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            dir=target_dir,
        )
        os.close(fd)
        try:
            torch.save(self.toDict(), temp_path)  # pyright: ignore[reportUnknownMemberType]
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

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
        evaluatorGeneratorState: Optional[torch.Tensor] = None,
        earlyStoppingState: Optional[Dict[str, Any]] = None,
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
            evaluatorGeneratorState=evaluatorGeneratorState,
            earlyStoppingState=earlyStoppingState,
        )


class CheckpointManager:
    def __init__(
        self,
        modelCfg: ModelConfig,
        trainCfg: TrainConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.modelCfg = modelCfg
        self.trainCfg = trainCfg
        self.ckptPath = trainCfg.ckptPath
        checkpointPath = Path(self.ckptPath)
        self.latestPath = str(checkpointPath.with_name("latest.pt"))
        self.checkpointDir = checkpointPath.parent
        self.logger = logger or logging.getLogger(__name__)

        ckptDir = os.path.dirname(self.ckptPath)
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
        evaluatorGeneratorState: Optional[torch.Tensor] = None,
        earlyStoppingState: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
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
            evaluatorGeneratorState=evaluatorGeneratorState,
            earlyStoppingState=earlyStoppingState,
            version=CHECKPOINT_VERSION,
        )
        checkpoint.save(path or self.ckptPath)

    def snapshotPath(self, step: int) -> str:
        return str(self.checkpointDir / f"step-{step:06d}.pt")

    def pruneSnapshots(self) -> None:
        snapshots = sorted(self.checkpointDir.glob("step-*.pt"))
        excess = len(snapshots) - self.trainCfg.maxSnapshots
        for snapshot in snapshots[:max(0, excess)]:
            snapshot.unlink()

    def resumePath(self) -> str:
        if os.path.exists(self.latestPath):
            return self.latestPath
        return self.ckptPath

    def loadCheckpoint(
        self,
        model: TinyGPTLanguageModel,
        optimizer: torch.optim.Optimizer,
        lrStrategy: Optional[Any] = None,
    ) -> Tuple[
        int,
        Optional[float],
        bool,
        int,
        bool,
        Dict[str, Dict[str, Any]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, Any]],
    ]:
        resumePath = self.resumePath()
        if not os.path.exists(resumePath):
            return 0, None, False, CHECKPOINT_VERSION, True, {}, None, None, None

        checkpoint = Checkpoint.load(resumePath, self.trainCfg.device)
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

        return (
            step,
            bestValLoss,
            lrStateRestored,
            version,
            version_matches,
            configDrift,
            generator_state,
            checkpoint.evaluatorGeneratorState,
            checkpoint.earlyStoppingState,
        )
