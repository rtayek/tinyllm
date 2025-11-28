# infer.py
from __future__ import annotations

import logging

from Config import RunConfig
from Model import TinyGpt
from Checkpoints import CheckpointManager
from TextGenerator import TextGenerator


def main() -> None:
    # Load config
    run_cfg = RunConfig()
    model_cfg = run_cfg.model
    train_cfg = run_cfg.train
    device = train_cfg.device

    # Build model
    model = TinyGpt(model_cfg).to(device)

    # Checkpoint manager
    ckpt_mgr = CheckpointManager(model_cfg, train_cfg)

    # Load model-only checkpoint (or fall back to training checkpoint)
    ckpt_mgr.loadModel(model, None)
    print("Model weights loaded for inference.")

    # Generate text
    logger = logging.getLogger("infer")
    text_gen = TextGenerator(model, train_cfg.device, logger)

    text = text_gen.generate_text(maxNewTokens=400)

    print("\n=== GENERATED TEXT ===\n")
    print(text)


if __name__ == "__main__":
    main()
