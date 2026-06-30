from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, cast

from llm.Config import ModelConfig, RunConfig, TrainConfig
from llm.corpus_sources import WORKS
from llm.train_app import buildTrainer, setupLogging, writeRunMetadata


@dataclass(frozen=True)
class RegressionCorpus:
    name: str
    trainPath: str
    validationPath: str
    testPath: str


@dataclass(frozen=True)
class RegressionJob:
    profile: str
    corpus: RegressionCorpus
    runDir: Path
    runConfig: RunConfig


@dataclass(frozen=True)
class RegressionResult:
    corpus: str
    runDir: Path
    bestValLoss: float
    checkpointPath: Path
    runJsonPath: Path
    metricsPath: Path


def _split_path(identifier: str, split: str, root: Path) -> str:
    return str(root / identifier / "splits" / f"{split}.txt")


def configured_corpora(root: Path = Path("corpora")) -> list[RegressionCorpus]:
    corpora = [
        RegressionCorpus(
            name=spec.identifier,
            trainPath=_split_path(spec.identifier, "train", root),
            validationPath=_split_path(spec.identifier, "validation", root),
            testPath=_split_path(spec.identifier, "test", root),
        )
        for spec in WORKS
    ]
    corpora.append(
        RegressionCorpus(
            name="jane-austen/combined",
            trainPath=_split_path("jane-austen/combined", "train", root),
            validationPath=_split_path("jane-austen/combined", "validation", root),
            testPath=_split_path("jane-austen/combined", "test", root),
        )
    )
    return corpora


def corpora_for_profile(profile: str) -> list[RegressionCorpus]:
    corpora = configured_corpora()
    if profile == "smoke":
        return corpora[:2]
    if profile == "all":
        return corpora
    raise ValueError(f"Unknown training regression profile: {profile}")


def _safe_name(name: str) -> str:
    return name.replace("/", "__")


def regression_model_config() -> ModelConfig:
    return ModelConfig(
        vocabSize=256,
        blockSize=16,
        nEmbed=32,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )


def regression_train_config(
    corpus: RegressionCorpus,
    runDir: Path,
    seed: int,
) -> TrainConfig:
    return TrainConfig(
        seed=seed,
        batchSize=2,
        learningRate=1e-3,
        warmupFrac=0.1,
        maxSteps=2,
        evalInterval=1,
        evalIters=1,
        snapshotInterval=0,
        maxSnapshots=0,
        weightDecay=0.0,
        earlyStopPatience=20,
        earlyStopDelta=0.0,
        plotCurve=False,
        dataModule="byte",
        runDir=str(runDir),
        ckptPath=str(runDir / "checkpoints" / "best.pt"),
        dataPath=corpus.trainPath,
        validationDataPath=corpus.validationPath,
        testDataPath=corpus.testPath,
        device="cpu",
    )


def plan_regression_jobs(profile: str) -> list[RegressionJob]:
    jobs: list[RegressionJob] = []
    for index, corpus in enumerate(corpora_for_profile(profile)):
        runDir = Path("runs") / "regression" / profile / _safe_name(corpus.name)
        trainConfig = regression_train_config(
            corpus,
            runDir,
            seed=2026 + index,
        )
        jobs.append(
            RegressionJob(
                profile=profile,
                corpus=corpus,
                runDir=runDir,
                runConfig=RunConfig(
                    modelConfig=regression_model_config(),
                    trainConfig=trainConfig,
                ),
            )
        )
    return jobs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual tinyllm training regressions.")
    parser.add_argument(
        "--profile",
        choices=("smoke", "all"),
        required=True,
        help="Regression profile to run.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete an existing regression run directory before running.",
    )
    return parser.parse_args(argv)


def _is_safe_regression_run_dir(path: Path) -> bool:
    parts = path.parts
    return len(parts) >= 4 and parts[0] == "runs" and parts[1] == "regression"


def _make_writable(path: str) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    except OSError:
        pass


def _rmtree_with_readonly_retry(path: Path) -> None:
    def onerror(
        func: Callable[[str], object],
        raw_path: str,
        _exc_info: object,
    ) -> None:
        _make_writable(raw_path)
        func(raw_path)

    shutil.rmtree(path, onerror=onerror)  # pyright: ignore[reportDeprecated]


def clear_regression_run_dir(path: Path) -> None:
    if not _is_safe_regression_run_dir(path):
        raise ValueError(
            f"Refusing to delete non-regression run directory: {path}. "
            "Expected a path under runs/regression/<profile>/<corpus-name>."
        )
    if not path.exists():
        return
    try:
        _rmtree_with_readonly_retry(path)
    except PermissionError as exc:
        raise PermissionError(
            f"Could not remove regression run directory {path}. On Windows, "
            "Explorer, an editor, antivirus, or another process may have a file "
            "or directory open. Manually remove this directory and rerun."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not remove regression run directory {path}. On Windows, "
            "Explorer, an editor, antivirus, or another process may have a file "
            "or directory open. Manually remove this directory and rerun."
        ) from exc


def _assert_run_artifacts(job: RegressionJob, bestValLoss: float) -> RegressionResult:
    if not math.isfinite(bestValLoss):
        raise RuntimeError(f"{job.corpus.name}: non-finite best validation loss")
    if bestValLoss >= 10.0:
        raise RuntimeError(
            f"{job.corpus.name}: best validation loss {bestValLoss:.4f} exceeds 10.0"
        )

    checkpointPath = Path(job.runConfig.trainConfig.ckptPath)
    runJsonPath = job.runDir / "run.json"
    metricsPath = job.runDir / "metrics.jsonl"
    for path in (checkpointPath, runJsonPath, metricsPath):
        if not path.exists():
            raise RuntimeError(f"{job.corpus.name}: expected artifact missing: {path}")

    metadata = cast(dict[str, Any], json.loads(runJsonPath.read_text(encoding="utf-8")))
    corpora = metadata.get("corpora")
    if not isinstance(corpora, dict):
        raise RuntimeError(f"{job.corpus.name}: run.json missing corpora metadata")
    corpora_dict = cast(dict[str, Any], corpora)
    train = corpora_dict.get("train")
    if not isinstance(train, dict):
        raise RuntimeError(f"{job.corpus.name}: run.json missing train corpus hash")
    train_dict = cast(dict[str, Any], train)
    if train_dict.get("sha256") is None:
        raise RuntimeError(f"{job.corpus.name}: run.json missing train corpus hash")

    return RegressionResult(
        corpus=job.corpus.name,
        runDir=job.runDir,
        bestValLoss=bestValLoss,
        checkpointPath=checkpointPath,
        runJsonPath=runJsonPath,
        metricsPath=metricsPath,
    )


def run_regression_job(
    job: RegressionJob,
    logger: logging.Logger,
    clean: bool = True,
) -> RegressionResult:
    if clean:
        clear_regression_run_dir(job.runDir)
    logger.info("Running training regression: %s -> %s", job.corpus.name, job.runDir)
    trainer = buildTrainer(job.runConfig, log=logger)
    trainer.loadCheckpointIfExists()
    writeRunMetadata(trainer, logger)
    trainingRan = trainer.train()
    if not trainingRan:
        raise RuntimeError(f"{job.corpus.name}: training did not run")
    trainer.evaluateBestCheckpointOnTest()
    if trainer.bestValLoss is None:
        raise RuntimeError(f"{job.corpus.name}: no validation loss recorded")
    return _assert_run_artifacts(job, trainer.bestValLoss)


def run_profile(
    profile: str,
    logger: logging.Logger | None = None,
    clean: bool = True,
) -> list[RegressionResult]:
    activeLogger = logger or logging.getLogger(__name__)
    return [
        run_regression_job(job, activeLogger, clean=clean)
        for job in plan_regression_jobs(profile)
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    activeLogger = setupLogging()
    results = run_profile(args.profile, activeLogger, clean=not args.no_clean)
    for result in results:
        print(
            f"{result.corpus}: best_val_loss={result.bestValLoss:.4f} "
            f"run_dir={result.runDir}"
        )


if __name__ == "__main__":
    main()
