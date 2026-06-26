from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from typing import Sequence

from llm.Config import ModelConfig, RunConfig, TrainConfig
from llm.Model import TinyGPTLanguageModel
from llm.Checkpoint import Checkpoint
from llm.TextGenerator import AutoregressiveGenerator
from llm.cli_utils import require_non_negative, require_positive
from llm.tensor_utils import resolve_device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with the tiny LLM")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the training checkpoint",
    )
    parser.add_argument("--prompt", default="", help="Text to use as the generation prompt")
    parser.add_argument(
        "--tokens",
        type=int,
        default=400,
        help="Number of new tokens to generate (default: 400)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Sample only from the K most likely tokens",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    args = parser.parse_args(argv)
    require_non_negative(parser, "--tokens", args.tokens)
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    if args.top_k is not None:
        require_positive(parser, "--top-k", args.top_k)
    if args.seed is not None:
        require_non_negative(parser, "--seed", args.seed)
    return args


def build_generator(
    run_cfg: RunConfig | None = None,
    logger: logging.Logger | None = None,
) -> tuple[AutoregressiveGenerator, TrainConfig]:
    run_cfg = run_cfg or RunConfig()
    model_cfg = run_cfg.modelConfig
    train_cfg = run_cfg.trainConfig
    active_logger = logger or logging.getLogger("infer")
    device = resolve_device(train_cfg.device, active_logger)
    train_cfg = replace(train_cfg, device=device)

    checkpoint = Checkpoint.load(train_cfg.ckptPath, device)
    if checkpoint.modelConfig:
        model_cfg = ModelConfig.fromDict(checkpoint.modelConfig)
    model = TinyGPTLanguageModel(model_cfg).to(device)
    model.load_state_dict(checkpoint.modelState)
    return AutoregressiveGenerator(model, device, active_logger), train_cfg


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_cfg = RunConfig()
    if args.checkpoint:
        run_cfg = RunConfig(
            modelConfig=run_cfg.modelConfig,
            trainConfig=replace(run_cfg.trainConfig, ckptPath=args.checkpoint),
        )
    textGenerator, _ = build_generator(run_cfg)
    print("Model weights loaded for inference.")

    text = textGenerator.generateText(
        maxNewTokens=args.tokens,
        prompt=args.prompt,
        temperature=args.temperature,
        topK=args.top_k,
        seed=args.seed,
    )

    print("\n=== GENERATED TEXT ===\n")
    sys.stdout.flush()
    sys.stdout.buffer.write((text + "\n").encode(sys.stdout.encoding or "utf-8", errors="replace"))


if __name__ == "__main__":
    main()
