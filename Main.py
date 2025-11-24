# Main.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import torch

from Config import ModelConfig
from DataModule import ByteDataModule
from Model import TinyGpt
from Trainer import Trainer


def main() -> None:
    torch.manual_seed(1337)

    cfg = ModelConfig()

    print("Loading data module...", flush=True)
    dataModule = ByteDataModule(cfg)

    print("Building model...", flush=True)
    model = TinyGpt(cfg).to(cfg.device)

    trainer = Trainer(cfg, model, dataModule)

    print("Loading checkpoint (if any)...", flush=True)
    trainer.loadCheckpointIfExists()

    trainer.train()
    trainer.plotTrainingCurve()
    trainer.printSample(maxNewTokens=200)


if __name__ == "__main__":
    main()
