# Main.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import torch

from Config import ModelConfig
from DataModule import ByteDataModule
from Model import TinyGpt
from Trainer import Trainer


def build_trainer(cfg: ModelConfig | None = None) -> Trainer:
    torch.manual_seed(1337)

    cfg = cfg or ModelConfig()

    print("Loading data module...")
    data_module = ByteDataModule(cfg)

    print("Building model...")
    model = TinyGpt(cfg).to(cfg.device)

    return Trainer(cfg, model, data_module)


def main() -> None:
    trainer = build_trainer()

    print("Loading checkpoint (if any)...")
    trainer.loadCheckpointIfExists()

    trainer.train()
    trainer.plotTrainingCurve()
    trainer.printSample(maxNewTokens=200)


if __name__ == "__main__":
    main()
