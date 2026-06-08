from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Sequence

from llm.Config import RunConfig, TrainConfig
from llm.Model import TinyGPTLanguageModel
from llm.Checkpoint import CheckpointManager
from llm.TextGenerator import AutoregressiveGenerator
from llm.tensor_utils import resolve_device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with the tiny LLM")
    parser.add_argument("--prompt", default="", help="Text to use as the generation prompt")
    parser.add_argument(
        "--tokens",
        type=int,
        default=400,
        help="Number of new tokens to generate (default: 400)",
    )
    args = parser.parse_args(argv)
    if args.tokens < 0:
        parser.error("--tokens must be non-negative")
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

    model = TinyGPTLanguageModel(model_cfg).to(device)

    checkpointManager = CheckpointManager(
        model_cfg,
        train_cfg,
        logger=active_logger,
    )

    checkpointManager.loadModel(model, None)
    return AutoregressiveGenerator(model, device, active_logger), train_cfg


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    textGenerator, _ = build_generator()
    print("Model weights loaded for inference.")

    text = textGenerator.generateText(maxNewTokens=args.tokens, prompt=args.prompt)

    print("\n=== GENERATED TEXT ===\n")
    print(text)


if __name__ == "__main__":
    main()
