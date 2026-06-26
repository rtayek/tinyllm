from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, cast
import logging
from dataclasses import dataclass, field

import torch

from .Config import ModelConfig, TrainConfig
from .Model import TinyGPTLanguageModel
from .serialization_types import (
    CheckpointPayload,
    CheckpointState,
    ConfigDrift,
    ConfigPayload,
)
from .tensor_utils import resolve_device

_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = "weights_only" in inspect.signature(cast(Any, torch.load)).parameters  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]


CHECKPOINT_VERSION = 1
_MODEL_SHAPE_FIELDS = frozenset(
    {"vocabSize", "blockSize", "nEmbed", "nHead", "nLayer"}
)


@dataclass
class CheckpointLoadResult:
    step: int = 0
    bestValLoss: Optional[float] = None
    lrStateRestored: bool = False
    version: int = CHECKPOINT_VERSION
    versionMatches: bool = True
    configDrift: ConfigDrift = field(default_factory=lambda: cast(ConfigDrift, {}))
    generatorState: Optional[torch.Tensor] = None
    evaluatorGeneratorState: Optional[torch.Tensor] = None
    earlyStoppingState: Optional[CheckpointPayload] = None


def _cpu_rng_state(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError("Checkpoint RNG state must be a tensor")
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


@dataclass
class Checkpoint:
    version: int
    modelState: CheckpointState
    optimizerState: CheckpointState
    step: int
    bestValLoss: Optional[float]
    modelConfig: ConfigPayload
    trainConfig: ConfigPayload
    lrStrategyState: Optional[CheckpointPayload] = None
    generatorState: Optional[torch.Tensor] = None
    evaluatorGeneratorState: Optional[torch.Tensor] = None
    earlyStoppingState: Optional[CheckpointPayload] = None

    def toDict(self) -> CheckpointPayload:
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
    def fromDict(data: CheckpointPayload) -> "Checkpoint":
        return Checkpoint(
            version=int(data.get("version", CHECKPOINT_VERSION)),
            modelState=cast(CheckpointState, data["modelState"]),
            optimizerState=cast(CheckpointState, data["optimizerState"]),
            step=int(data.get("step", 0)),
            bestValLoss=data.get("bestValLoss", None),
            modelConfig=cast(ConfigPayload, data.get("modelConfig", {})),
            trainConfig=cast(ConfigPayload, data.get("trainConfig", {})),
            lrStrategyState=cast(Optional[CheckpointPayload], data.get("lrStrategyState", None)),
            generatorState=_cpu_rng_state(data.get("generatorState", None)),
            evaluatorGeneratorState=_cpu_rng_state(
                data.get("evaluatorGeneratorState", None)
            ),
            earlyStoppingState=cast(
                Optional[CheckpointPayload],
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
        resolved_device = resolve_device(str(device), logging.getLogger(__name__))
        if _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
            data = cast(CheckpointPayload, torch.load(path, map_location=resolved_device, weights_only=False))  # pyright: ignore[reportUnknownMemberType]
        else:
            data = cast(CheckpointPayload, torch.load(path, map_location=resolved_device))  # pyright: ignore[reportUnknownMemberType]
        return Checkpoint.fromDict(data)

    @staticmethod
    def fromTrainingState(
        model: TinyGPTLanguageModel,
        optimizer: torch.optim.Optimizer,
        modelConfig: Optional[ModelConfig],
        trainConfig: Optional[TrainConfig],
        step: int,
        bestValLoss: Optional[float],
        lrStrategyState: Optional[CheckpointPayload] = None,
        generatorState: Optional[torch.Tensor] = None,
        evaluatorGeneratorState: Optional[torch.Tensor] = None,
        earlyStoppingState: Optional[CheckpointPayload] = None,
        version: int = CHECKPOINT_VERSION,
    ) -> "Checkpoint":
        return Checkpoint(
            version=version,
            modelState=model.state_dict(),
            optimizerState=optimizer.state_dict(),
            step=step,
            bestValLoss=bestValLoss,
            modelConfig=modelConfig.toDict() if modelConfig is not None else {},
            trainConfig=trainConfig.toDict() if trainConfig is not None else {},
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
        lrStrategyState: Optional[CheckpointPayload],
        step: int,
        bestValLoss: Optional[float],
        generatorState: Optional[torch.Tensor] = None,
        evaluatorGeneratorState: Optional[torch.Tensor] = None,
        earlyStoppingState: Optional[CheckpointPayload] = None,
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
    ) -> CheckpointLoadResult:
        resumePath = self.resumePath()
        if not os.path.exists(resumePath):
            return CheckpointLoadResult()

        checkpoint = Checkpoint.load(resumePath, self.trainCfg.device)
        if checkpoint.modelConfig:
            currentModelDict = self.modelCfg.toDict()
            incompatible = {
                k: (currentModelDict[k], v)
                for k, v in checkpoint.modelConfig.items()
                if (
                    k in _MODEL_SHAPE_FIELDS
                    and k in currentModelDict
                    and currentModelDict[k] != v
                )
            }
            if incompatible:
                details = ", ".join(
                    f"{key}: current={current!r}, checkpoint={saved!r}"
                    for key, (current, saved) in sorted(incompatible.items())
                )
                raise ValueError(
                    "Checkpoint model config is incompatible with the requested "
                    f"model config at {resumePath}: {details}. Use a new "
                    "--run-dir/--checkpoint or restart from scratch."
                )
        model.load_state_dict(checkpoint.modelState)
        optimizer.load_state_dict(checkpoint.optimizerState)

        version = checkpoint.version
        version_matches = version == CHECKPOINT_VERSION

        lrStateRestored = False
        if lrStrategy is not None and version_matches:
            schedState = checkpoint.lrStrategyState
            if schedState is not None:
                lrStrategy.load_state_dict(schedState)
                lrStateRestored = True

        configDrift: ConfigDrift = {}
        savedModelConfig = checkpoint.modelConfig
        savedTrainConfig = checkpoint.trainConfig
        if savedModelConfig:
            currentModelDict = self.modelCfg.toDict()
            configDrift["model"] = {
                k: v
                for k, v in savedModelConfig.items()
                if k in currentModelDict and currentModelDict[k] != v
            }
        if savedTrainConfig:
            currentTrainDict = self.trainCfg.toDict()
            configDrift["train"] = {
                k: v
                for k, v in savedTrainConfig.items()
                if k in currentTrainDict and currentTrainDict[k] != v
            }

        return CheckpointLoadResult(
            step=checkpoint.step,
            bestValLoss=checkpoint.bestValLoss,
            lrStateRestored=lrStateRestored,
            version=version,
            versionMatches=version_matches,
            configDrift=configDrift,
            generatorState=checkpoint.generatorState,
            evaluatorGeneratorState=checkpoint.evaluatorGeneratorState,
            earlyStoppingState=checkpoint.earlyStoppingState,
        )
