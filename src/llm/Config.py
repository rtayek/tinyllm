from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, cast


@dataclass(frozen=True)
class ModelConfig:
    vocabSize: int = 256
    blockSize: int = 128
    nEmbed: int = 256
    nHead: int = 4
    nLayer: int = 4
    dropout: float = 0.2
    use_cache: bool = False

    def __post_init__(self) -> None:
        if self.blockSize < 1:
            raise ValueError(f"blockSize must be >= 1, got {self.blockSize}")
        if self.vocabSize < 1:
            raise ValueError(f"vocabSize must be >= 1, got {self.vocabSize}")

    def toDict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1337
    batchSize: int = 32
    learningRate: float = 5e-5
    warmupFrac: float = 0.1
    maxSteps: int = 5000
    evalInterval: int = 100
    evalIters: int = 100
    snapshotInterval: int = 1000
    maxSnapshots: int = 3
    weightDecay: float = 0.02
    earlyStopPatience: int = 2
    earlyStopDelta: float = 0.003
    plotCurve: bool = True
    dataModule: str = "token"
    ckptPath: str = "runs/sherlock-byte-default/checkpoints/best.pt"
    dataPath: str = (
        "corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/"
        "splits/train.txt"
    )
    validationDataPath: str | None = (
        "corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/"
        "splits/validation.txt"
    )
    testDataPath: str | None = (
        "corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/"
        "splits/test.txt"
    )
    device: str = "cuda"  # desired/default device; actual availability is checked at runtime

    def __post_init__(self) -> None:
        if self.batchSize < 1:
            raise ValueError(f"batchSize must be >= 1, got {self.batchSize}")
        if self.learningRate <= 0:
            raise ValueError(f"learningRate must be > 0, got {self.learningRate}")
        if not (0 <= self.warmupFrac <= 1):
            raise ValueError(f"warmupFrac must be in [0, 1], got {self.warmupFrac}")
        if self.evalIters < 1:
            raise ValueError(f"evalIters must be >= 1, got {self.evalIters}")

    def runDirectory(self) -> Path | None:
        checkpointDir = Path(self.ckptPath).parent
        if checkpointDir.name == "checkpoints" and checkpointDir.parent != Path("."):
            return checkpointDir.parent
        return None

    _PATH_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"dataPath", "validationDataPath", "testDataPath"}
    )

    def toDict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def toSerializableDict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in self._PATH_FIELDS}

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "TrainConfig":
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RunConfig:
    modelConfig: ModelConfig = ModelConfig()
    trainConfig: TrainConfig = TrainConfig()
    def toDict(self) -> Dict[str, Any]:
        return {"model": self.modelConfig.toDict(), "train": self.trainConfig.toDict()}

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "RunConfig":
        modelData = data.get("model", {})
        trainData = data.get("train", {})
        modelConfig = ModelConfig.fromDict(cast(Dict[str, Any], modelData)) if isinstance(modelData, dict) else ModelConfig()
        trainConfig = TrainConfig.fromDict(cast(Dict[str, Any], trainData)) if isinstance(trainData, dict) else TrainConfig()
        return cls(modelConfig=modelConfig, trainConfig=trainConfig)
