from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

from llm.serialization_types import ConfigPayload


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
        if self.nEmbed < 1:
            raise ValueError(f"nEmbed must be >= 1, got {self.nEmbed}")
        if self.nHead < 1:
            raise ValueError(f"nHead must be >= 1, got {self.nHead}")
        if self.nLayer < 1:
            raise ValueError(f"nLayer must be >= 1, got {self.nLayer}")
        if self.nEmbed % self.nHead != 0:
            raise ValueError(
                f"nEmbed must be divisible by nHead, got {self.nEmbed} and {self.nHead}"
            )
        if not (0 <= self.dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def toDict(self) -> ConfigPayload:
        return dict(self.__dict__)

    @classmethod
    def fromDict(cls, data: ConfigPayload) -> "ModelConfig":
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
    earlyStopPatience: int = 10
    earlyStopDelta: float = 0.001
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
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.batchSize < 1:
            raise ValueError(f"batchSize must be >= 1, got {self.batchSize}")
        if self.learningRate <= 0:
            raise ValueError(f"learningRate must be > 0, got {self.learningRate}")
        if not (0 <= self.warmupFrac <= 1):
            raise ValueError(f"warmupFrac must be in [0, 1], got {self.warmupFrac}")
        if self.maxSteps < 1:
            raise ValueError(f"maxSteps must be >= 1, got {self.maxSteps}")
        if self.evalInterval < 1:
            raise ValueError(f"evalInterval must be >= 1, got {self.evalInterval}")
        if self.evalIters < 1:
            raise ValueError(f"evalIters must be >= 1, got {self.evalIters}")
        if self.snapshotInterval < 0:
            raise ValueError(
                f"snapshotInterval must be >= 0, got {self.snapshotInterval}"
            )
        if self.maxSnapshots < 0:
            raise ValueError(f"maxSnapshots must be >= 0, got {self.maxSnapshots}")
        if self.weightDecay < 0:
            raise ValueError(f"weightDecay must be >= 0, got {self.weightDecay}")
        if self.earlyStopPatience < 1:
            raise ValueError(
                f"earlyStopPatience must be >= 1, got {self.earlyStopPatience}"
            )
        if self.earlyStopDelta < 0:
            raise ValueError(
                f"earlyStopDelta must be >= 0, got {self.earlyStopDelta}"
            )

    def runDirectory(self) -> Path | None:
        checkpointDir = Path(self.ckptPath).parent
        if checkpointDir.name == "checkpoints" and checkpointDir.parent != Path("."):
            return checkpointDir.parent
        return None

    _PATH_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"dataPath", "validationDataPath", "testDataPath"}
    )

    def toDict(self) -> ConfigPayload:
        return dict(self.__dict__)

    def toSerializableDict(self) -> ConfigPayload:
        return {k: v for k, v in self.__dict__.items() if k not in self._PATH_FIELDS}

    @classmethod
    def fromDict(cls, data: ConfigPayload) -> "TrainConfig":
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RunConfig:
    modelConfig: ModelConfig = ModelConfig()
    trainConfig: TrainConfig = TrainConfig()
    def toDict(self) -> ConfigPayload:
        return {"model": self.modelConfig.toDict(), "train": self.trainConfig.toDict()}

    @classmethod
    def fromDict(cls, data: ConfigPayload) -> "RunConfig":
        modelData = data.get("model", {})
        trainData = data.get("train", {})
        if not isinstance(modelData, dict):
            raise ValueError(f"Expected 'model' to be a dict, got {type(modelData).__name__}")
        if not isinstance(trainData, dict):
            raise ValueError(f"Expected 'train' to be a dict, got {type(trainData).__name__}")
        modelConfig = ModelConfig.fromDict(cast(ConfigPayload, modelData)) if modelData else ModelConfig()
        trainConfig = TrainConfig.fromDict(cast(ConfigPayload, trainData)) if trainData else TrainConfig()
        return cls(modelConfig=modelConfig, trainConfig=trainConfig)

    @classmethod
    def fromRunJson(cls, path: str | Path) -> "RunConfig":
        rawData: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(rawData, dict):
            raise ValueError(f"Expected run metadata to be a dict, got {type(rawData).__name__}")
        data = cast(ConfigPayload, rawData)

        modelData = data.get("model", {})
        trainData = data.get("training", data.get("train", {}))
        corporaData = data.get("corpora", {})
        if not isinstance(modelData, dict):
            raise ValueError(f"Expected 'model' to be a dict, got {type(modelData).__name__}")
        if not isinstance(trainData, dict):
            raise ValueError(f"Expected 'training' to be a dict, got {type(trainData).__name__}")
        if not isinstance(corporaData, dict):
            raise ValueError(f"Expected 'corpora' to be a dict, got {type(corporaData).__name__}")

        replayTrainData = dict(cast(ConfigPayload, trainData))
        for splitName, fieldName in (
            ("train", "dataPath"),
            ("validation", "validationDataPath"),
            ("test", "testDataPath"),
        ):
            splitData = cast(ConfigPayload, corporaData).get(splitName)
            if isinstance(splitData, dict) and "path" in splitData:
                replayTrainData[fieldName] = splitData["path"]

        modelConfig = ModelConfig.fromDict(cast(ConfigPayload, modelData)) if modelData else ModelConfig()
        trainConfig = TrainConfig.fromDict(replayTrainData) if replayTrainData else TrainConfig()
        return cls(modelConfig=modelConfig, trainConfig=trainConfig)
