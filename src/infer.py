from __future__ import annotations

import logging

from Config import RunConfig
from Model import TinyGpt
from Checkpoint import CheckpointManager
from TextGenerator import TextGenerator


def main() -> None:
    run_cfg = RunConfig()
    model_cfg = run_cfg.model
    train_cfg = run_cfg.train
    device = train_cfg.device

    model = TinyGpt(model_cfg).to(device)

    logger = logging.getLogger("infer")
    checkpointManager = CheckpointManager(model_cfg, train_cfg, logger=logger)

    checkpointManager.loadModel(model, None)
    print("Model weights loaded for inference.")

    textGenerator = TextGenerator(model, train_cfg.device, logger)

    text = textGenerator.generateText(maxNewTokens=400)

    print("\n=== GENERATED TEXT ===\n")
    print(text)


if __name__ == "__main__":
    main()
