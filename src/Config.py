from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, cast
import torch


@dataclass(frozen=True)
class ModelConfig:
    vocabSize: int = 256
    blockSize: int = 128
    nEmbed: int = 256
    nHead: int = 4
    nLayer: int = 4
    dropout: float = 0.2
    def toDict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrainConfig:
    batchSize: int = 32
    learningRate: float = 5e-5
    warmupFrac: float = 0.1
    maxSteps: int = 5000
    evalInterval: int = 100
    evalIters: int = 100
    weightDecay: float = 0.02
    earlyStopPatience: int = 2
    earlyStopDelta: float = 0.003
    plotCurve: bool = True
    def toDict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "TrainConfig":
        return cls(**data)  # type: ignore[arg-type]

    ckptPath: str = "checkpoints/tiny_llm.pt"
    dataPath: str = "fixtureData/input.txt"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    def toDict(self) -> Dict[str, Any]:
        return {"model": self.model.toDict(), "train": self.train.toDict()}

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "RunConfig":
        modelData = data.get("model", {})
        trainData = data.get("train", {})
        modelConfig = ModelConfig.fromDict(cast(Dict[str, Any], modelData)) if isinstance(modelData, dict) else ModelConfig()
        trainConfig = TrainConfig.fromDict(cast(Dict[str, Any], trainData)) if isinstance(trainData, dict) else TrainConfig()
        return cls(model=modelConfig, train=trainConfig)
