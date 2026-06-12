import json
from pathlib import Path

from llm.Config import ModelConfig, TrainConfig
from llm.RunArtifacts import RunArtifacts


def test_run_artifacts_write_metadata_and_metrics(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    test = tmp_path / "test.txt"
    train.write_text("train", encoding="utf-8")
    validation.write_text("validation", encoding="utf-8")
    test.write_text("test", encoding="utf-8")
    run = tmp_path / "runs" / "experiment"
    config = TrainConfig(
        ckptPath=str(run / "checkpoints" / "best.pt"),
        dataPath=str(train),
        validationDataPath=str(validation),
        testDataPath=str(test),
        device="cpu",
    )
    artifacts = RunArtifacts(ModelConfig(), config)

    metadata_path = artifacts.writeRunMetadata()
    metrics_path = artifacts.appendMetric(
        {"type": "evaluation", "step": 10, "validation_loss": 1.5}
    )

    assert metadata_path == run / "run.json"
    assert metrics_path == run / "metrics.jsonl"
    assert metadata_path is not None
    assert metrics_path is not None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["training"]["ckptPath"] == str(
        run / "checkpoints" / "best.pt"
    )
    assert "dataPath" not in metadata["training"]
    assert "validationDataPath" not in metadata["training"]
    assert "testDataPath" not in metadata["training"]
    assert metadata["corpora"]["train"]["path"] == str(train)
    assert metadata["corpora"]["validation"]["path"] == str(validation)
    assert metadata["corpora"]["test"]["path"] == str(test)
    assert metadata["corpora"]["train"]["sha256"] is not None
    metric = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metric["type"] == "evaluation"
    assert metric["step"] == 10


def test_run_artifacts_ignore_legacy_checkpoint_path(tmp_path: Path) -> None:
    config = TrainConfig(
        ckptPath=str(tmp_path / "checkpoint.pt"),
        device="cpu",
    )
    artifacts = RunArtifacts(ModelConfig(), config)

    assert artifacts.writeRunMetadata() is None
    assert artifacts.appendMetric({"type": "test"}) is None
