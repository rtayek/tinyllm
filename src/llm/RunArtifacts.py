from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from .Config import ModelConfig, TrainConfig


class RunArtifacts:
    def __init__(self, modelConfig: ModelConfig, trainConfig: TrainConfig) -> None:
        self.modelConfig = modelConfig
        self.trainConfig = trainConfig
        self.runDirectory = trainConfig.runDirectory()

    @staticmethod
    def _sha256(path: str | None) -> str | None:
        if path is None:
            return None
        source = Path(path)
        if not source.exists():
            return None
        return hashlib.sha256(source.read_bytes()).hexdigest()

    @staticmethod
    def _git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None

    def _corporaPayload(self) -> dict[str, dict[str, str | None]]:
        return {
            "train": {
                "path": self.trainConfig.dataPath,
                "sha256": self._sha256(self.trainConfig.dataPath),
            },
            "validation": {
                "path": self.trainConfig.validationDataPath,
                "sha256": self._sha256(self.trainConfig.validationDataPath),
            },
            "test": {
                "path": self.trainConfig.testDataPath,
                "sha256": self._sha256(self.trainConfig.testDataPath),
            },
        }

    def _configPayload(self) -> dict[str, object]:
        return {
            "git_commit": self._git_commit(),
            "model": self.modelConfig.toDict(),
            "training": self.trainConfig.toSerializableDict(),
            "corpora": self._corporaPayload(),
        }

    def writeRunMetadata(self) -> Path | None:
        if self.runDirectory is None:
            return None
        self.runDirectory.mkdir(parents=True, exist_ok=True)
        path = self.runDirectory / "run.json"
        if path.exists():
            # Preserve the original run record on resume; don't overwrite
            # git_commit, created_at, or parent_checkpoint.
            return path
        checkpointPath = Path(self.trainConfig.ckptPath)
        latestPath = checkpointPath.with_name("latest.pt")
        parentCheckpoint = latestPath if latestPath.exists() else checkpointPath
        payload = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent_checkpoint": (
                {
                    "path": str(parentCheckpoint),
                    "sha256": self._sha256(str(parentCheckpoint)),
                }
                if parentCheckpoint.exists()
                else None
            ),
            **self._configPayload(),
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def runMetadataExists(self) -> bool:
        return self.runDirectory is not None and (self.runDirectory / "run.json").exists()

    def appendMetric(self, record: dict[str, object]) -> Path | None:
        if self.runDirectory is None:
            return None
        self.runDirectory.mkdir(parents=True, exist_ok=True)
        path = self.runDirectory / "metrics.jsonl"
        payload = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(payload, sort_keys=True) + "\n")
        return path

    def appendContinuation(self) -> Path | None:
        if self.runDirectory is None:
            return None
        runMetadataPath = self.runDirectory / "run.json"
        if not runMetadataPath.exists():
            return None
        self.runDirectory.mkdir(parents=True, exist_ok=True)
        path = self.runDirectory / "continuations.jsonl"
        payload = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **self._configPayload(),
        }
        with path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(payload, sort_keys=True) + "\n")
        return path
