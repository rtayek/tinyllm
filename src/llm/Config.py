from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, cast

from llm.serialization_types import ConfigPayload
from llm.json_utils import read_json


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError(message)`` unless ``condition`` holds.

    A thin guard so config validation reads as a list of declarative
    constraints rather than repeated ``if ...: raise`` blocks. Works uniformly
    for simple bound checks and cross-field checks (e.g. divisibility).
    """
    if not condition:
        raise ValueError(message)


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
        _require(self.blockSize >= 1, f"blockSize must be >= 1, got {self.blockSize}")
        _require(self.vocabSize >= 1, f"vocabSize must be >= 1, got {self.vocabSize}")
        _require(self.nEmbed >= 1, f"nEmbed must be >= 1, got {self.nEmbed}")
        _require(self.nHead >= 1, f"nHead must be >= 1, got {self.nHead}")
        _require(self.nLayer >= 1, f"nLayer must be >= 1, got {self.nLayer}")
        _require(
            self.nEmbed % self.nHead == 0,
            f"nEmbed must be divisible by nHead, got {self.nEmbed} and {self.nHead}",
        )
        _require(0 <= self.dropout < 1, f"dropout must be in [0, 1), got {self.dropout}")

    def toDict(self) -> ConfigPayload:
        return dict(self.__dict__)

    @classmethod
    def fromDict(cls, data: ConfigPayload) -> "ModelConfig":
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RunPaths:
    """The filesystem locations a run reads from and writes to.

    This is a narrowing *view* over the path-valued fields of ``TrainConfig``
    (mirroring ``DataModuleConfig.fromTrainConfig``). It groups the "where it
    ran" plumbing so consumers that only need paths can depend on this instead
    of reaching into the full ``TrainConfig``. ``TrainConfig`` remains the
    serialization boundary, so run.json and checkpoint formats are unchanged.
    """
    dataPath: str
    validationDataPath: str | None
    testDataPath: str | None
    ckptPath: str

    @classmethod
    def fromTrainConfig(cls, trainConfig: "TrainConfig") -> "RunPaths":
        return cls(
            dataPath=trainConfig.dataPath,
            validationDataPath=trainConfig.validationDataPath,
            testDataPath=trainConfig.testDataPath,
            ckptPath=trainConfig.ckptPath,
        )

    def runDirectory(self) -> Path | None:
        checkpointDir = Path(self.ckptPath).parent
        if checkpointDir.name == "checkpoints" and checkpointDir.parent != Path("."):
            return checkpointDir.parent
        return None


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
        _require(self.seed >= 0, f"seed must be >= 0, got {self.seed}")
        _require(self.batchSize >= 1, f"batchSize must be >= 1, got {self.batchSize}")
        _require(self.learningRate > 0, f"learningRate must be > 0, got {self.learningRate}")
        _require(0 <= self.warmupFrac <= 1, f"warmupFrac must be in [0, 1], got {self.warmupFrac}")
        _require(self.maxSteps >= 1, f"maxSteps must be >= 1, got {self.maxSteps}")
        _require(self.evalInterval >= 1, f"evalInterval must be >= 1, got {self.evalInterval}")
        _require(self.evalIters >= 1, f"evalIters must be >= 1, got {self.evalIters}")
        _require(
            self.snapshotInterval >= 0,
            f"snapshotInterval must be >= 0, got {self.snapshotInterval}",
        )
        _require(self.maxSnapshots >= 0, f"maxSnapshots must be >= 0, got {self.maxSnapshots}")
        _require(self.weightDecay >= 0, f"weightDecay must be >= 0, got {self.weightDecay}")
        _require(
            self.earlyStopPatience >= 1,
            f"earlyStopPatience must be >= 1, got {self.earlyStopPatience}",
        )
        _require(
            self.earlyStopDelta >= 0,
            f"earlyStopDelta must be >= 0, got {self.earlyStopDelta}",
        )

    def runDirectory(self) -> Path | None:
        return self.paths().runDirectory()

    def paths(self) -> "RunPaths":
        return RunPaths.fromTrainConfig(self)

    _PATH_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"dataPath", "validationDataPath", "testDataPath"}
    )

    def toDict(self) -> ConfigPayload:
        return dict(self.__dict__)

    def toRunJsonDict(self) -> ConfigPayload:
        return {k: v for k, v in self.__dict__.items() if k not in self._PATH_FIELDS}

    @classmethod
    def fromDict(cls, data: ConfigPayload) -> "TrainConfig":
        valid_fields = {field.name for field in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)  # type: ignore[arg-type]


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
        rawData: Any = read_json(Path(path))
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

        modelPayload = cast(ConfigPayload, modelData)
        trainPayload = cast(ConfigPayload, trainData)
        corporaPayload = cast(ConfigPayload, corporaData)

        modelConfig = ModelConfig.fromDict(modelPayload) if modelPayload else ModelConfig()
        trainConfig = cls._trainConfigFromRunJson(trainPayload, corporaPayload)
        return cls(modelConfig=modelConfig, trainConfig=trainConfig)

    @staticmethod
    def _trainConfigFromRunJson(
        trainData: ConfigPayload,
        corporaData: ConfigPayload,
    ) -> TrainConfig:
        replayTrainData = dict(trainData)
        for splitName, fieldName in (
            ("train", "dataPath"),
            ("validation", "validationDataPath"),
            ("test", "testDataPath"),
        ):
            splitData = corporaData.get(splitName)
            if isinstance(splitData, dict) and "path" in splitData:
                replayTrainData[fieldName] = splitData["path"]
        return TrainConfig.fromDict(replayTrainData) if replayTrainData else TrainConfig()
