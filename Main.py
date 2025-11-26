# Main.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import torch

from Config import ModelConfig, TrainConfig
from DataModule import ByteDataModule
from Model import TinyGpt
from Trainer import Trainer


def build_trainer(
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> Trainer:
    torch.manual_seed(1337)

    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    print("Loading data module...")
    data_module = ByteDataModule(model_cfg, train_cfg)

    print("Building model...")
    model = TinyGpt(model_cfg).to(train_cfg.device)

    return Trainer(model_cfg, train_cfg, model, data_module)


def main() -> None:
    trainer = build_trainer()

    print("Loading checkpoint (if any)...")
    trainer.loadCheckpointIfExists()

    trainer.train()
    trainer.plotTrainingCurve()
    trainer.printSample(maxNewTokens=200)


if __name__ == "__main__":
    main()
