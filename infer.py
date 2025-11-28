# infer.py
from __future__ import annotations

import logging
import torch

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

    # Dummy optimizer (needed for checkpoint loading)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learningRate,
        weight_decay=train_cfg.weightDecay,
    )

    # Checkpoint manager
    ckpt_mgr = CheckpointManager(model_cfg, train_cfg)

    # Load checkpoint (no LR strategy needed for inference)
    step, best_val, _lr_restored, version, version_matches, config_drift = ckpt_mgr.load(
        model=model,
        optimizer=optimizer,
        lrStrategy=None,
    )

    print(f"Loaded checkpoint from step {step}, best val loss = {best_val}")
    print(f"Checkpoint version: {version} (matches expected: {version_matches})")

    # Config drift reporting
    if config_drift.get("model"):
        print("Model config drift:", config_drift["model"])
    if config_drift.get("train"):
        print("Train config drift:", config_drift["train"])

    # Generate text
    logger = logging.getLogger("infer")
    text_gen = TextGenerator(model, train_cfg.device, logger)

    text = text_gen.generate_text(maxNewTokens=400)

    print("\n=== GENERATED TEXT ===\n")
    print(text)


if __name__ == "__main__":
    main()
