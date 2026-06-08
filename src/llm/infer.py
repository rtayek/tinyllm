from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Sequence

from llm.Config import ModelConfig, RunConfig, TrainConfig
from llm.Model import TinyGPTLanguageModel
from llm.Checkpoint import Checkpoint
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
    if args.tokens < 0:
        parser.error("--tokens must be non-negative")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
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
    textGenerator, _ = build_generator()
    print("Model weights loaded for inference.")

    text = textGenerator.generateText(
        maxNewTokens=args.tokens,
        prompt=args.prompt,
        temperature=args.temperature,
        topK=args.top_k,
        seed=args.seed,
    )

    print("\n=== GENERATED TEXT ===\n")
    print(text)


if __name__ == "__main__":
    main()
