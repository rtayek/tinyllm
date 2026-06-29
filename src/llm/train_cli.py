from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple, Sequence

from .Config import RunConfig
from .cli_utils import parse_log_level, require_non_negative, require_positive


class TrainCliConfig(NamedTuple):
    runConfig: RunConfig
    resetEarlyStopping: bool
    logLevel: int


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the tiny LLM")
    parser.add_argument("--corpus", type=str, default=None, help="Path to training corpus (overrides TrainConfig.dataPath)")
    parser.add_argument("--validation-corpus", type=str, default=None, help="Path to validation corpus")
    parser.add_argument("--test-corpus", type=str, default=None, help="Path to test corpus")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to training checkpoint")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory; sets checkpoint to RUN_DIR/checkpoints/best.pt")
    parser.add_argument("--seed", type=int, default=None, help="Training random seed")
    parser.add_argument("--block-size", type=int, default=None, help="Transformer context window size")
    parser.add_argument("--n-embed", type=int, default=None, help="Transformer embedding width")
    parser.add_argument("--n-head", type=int, default=None, help="Transformer attention heads")
    parser.add_argument("--n-layer", type=int, default=None, help="Transformer decoder layers")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum optimizer steps to train")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="Evaluations without significant improvement before stopping")
    parser.add_argument("--reset-early-stopping", action="store_true", help="Reset restored early-stopping progress when resuming")
    parser.add_argument("--snapshot-interval", type=int, default=None, help="Steps between retained checkpoint snapshots")
    parser.add_argument("--max-snapshots", type=int, default=None, help="Maximum retained step snapshots")
    parser.add_argument("--plot", action="store_true", help="Enable plotting the training curve")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser


def runConfigFromArgs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    defaultLogLevel: int = logging.INFO,
) -> TrainCliConfig:
    level = parse_log_level(parser, args.log_level, defaultLogLevel)
    runConfig = RunConfig()
    modelConfig = runConfig.modelConfig
    trainConfig = runConfig.trainConfig

    if args.corpus:
        trainConfig = replace(trainConfig, dataPath=args.corpus)
    if args.validation_corpus:
        trainConfig = replace(trainConfig, validationDataPath=args.validation_corpus)
    if args.test_corpus:
        trainConfig = replace(trainConfig, testDataPath=args.test_corpus)
    if args.checkpoint:
        trainConfig = replace(trainConfig, ckptPath=args.checkpoint)
    if args.run_dir:
        trainConfig = replace(
            trainConfig,
            runDir=args.run_dir,
            ckptPath=str(Path(args.run_dir) / "checkpoints" / "best.pt"),
        )
    if args.block_size is not None:
        require_positive(parser, "--block-size", args.block_size)
        modelConfig = replace(modelConfig, blockSize=args.block_size)
    if args.n_embed is not None:
        require_positive(parser, "--n-embed", args.n_embed)
        modelConfig = replace(modelConfig, nEmbed=args.n_embed)
    if args.n_head is not None:
        require_positive(parser, "--n-head", args.n_head)
        modelConfig = replace(modelConfig, nHead=args.n_head)
    if args.n_layer is not None:
        require_positive(parser, "--n-layer", args.n_layer)
        modelConfig = replace(modelConfig, nLayer=args.n_layer)
    if args.seed is not None:
        require_non_negative(parser, "--seed", args.seed)
        trainConfig = replace(trainConfig, seed=args.seed)
    if args.max_steps is not None:
        require_positive(parser, "--max-steps", args.max_steps)
        trainConfig = replace(trainConfig, maxSteps=args.max_steps)
    if args.early_stop_patience is not None:
        require_positive(parser, "--early-stop-patience", args.early_stop_patience)
        trainConfig = replace(
            trainConfig,
            earlyStopPatience=args.early_stop_patience,
        )
    if args.snapshot_interval is not None:
        require_non_negative(parser, "--snapshot-interval", args.snapshot_interval)
        trainConfig = replace(trainConfig, snapshotInterval=args.snapshot_interval)
    if args.max_snapshots is not None:
        require_non_negative(parser, "--max-snapshots", args.max_snapshots)
        trainConfig = replace(trainConfig, maxSnapshots=args.max_snapshots)
    if args.plot:
        trainConfig = replace(trainConfig, plotCurve=True)
    return TrainCliConfig(
        runConfig=RunConfig(modelConfig=modelConfig, trainConfig=trainConfig),
        resetEarlyStopping=args.reset_early_stopping,
        logLevel=level,
    )


def parseTrainCli(
    argv: Sequence[str] | None = None,
    defaultLogLevel: int = logging.INFO,
) -> TrainCliConfig:
    parser = buildParser()
    args = parser.parse_args(argv)
    return runConfigFromArgs(args, parser, defaultLogLevel=defaultLogLevel)
