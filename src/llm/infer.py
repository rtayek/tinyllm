from __future__ import annotations

import argparse
import logging
from typing import Sequence

from llm.Config import RunConfig
from llm.Model import TinyGPTLanguageModel
from llm.Checkpoint import CheckpointManager
from llm.TextGenerator import AutoregressiveGenerator


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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_cfg = RunConfig()
    model_cfg = run_cfg.modelConfig
    train_cfg = run_cfg.trainConfig
    device = train_cfg.device

    model = TinyGPTLanguageModel(model_cfg).to(device)

    logger = logging.getLogger("infer")
    checkpointManager = CheckpointManager(model_cfg, train_cfg, logger=logger)

    checkpointManager.loadModel(model, None)
    print("Model weights loaded for inference.")

    textGenerator = AutoregressiveGenerator(model, train_cfg.device, logger)

    text = textGenerator.generateText(maxNewTokens=args.tokens, prompt=args.prompt)

    print("\n=== GENERATED TEXT ===\n")
    print(text)


if __name__ == "__main__":
    main()
