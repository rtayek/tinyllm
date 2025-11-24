# DataModule.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
from typing import Tuple, List, Optional

import torch
from torch import Tensor

from Config import ModelConfig


class ByteDataModule:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.trainData: Tensor
        self.valData: Tensor

        self.loadAndSplit()

    def loadAndSplit(self) -> None:
        if not os.path.exists(self.cfg.dataPath):
            raise FileNotFoundError(self.cfg.dataPath)

        with open(self.cfg.dataPath, "rb") as f:
            dataBytes = f.read()

        data = torch.tensor(list(dataBytes), dtype=torch.long)
        boundary = int(0.9 * len(data))
        self.trainData = data[:boundary]
        self.valData = data[boundary:]

    def getBatch(
        self,
        split: str,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        if split == "train":
            source = self.trainData
        elif split == "val":
            source = self.valData
        else:
            raise ValueError(f"Unknown split: {split}")

        cfg = self.cfg

        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(1337)

        indices = torch.randint(
            low=0,
            high=len(source) - cfg.blockSize - 1,
            size=(cfg.batchSize,),
            generator=generator,
        )

        xList: List[Tensor] = []
        yList: List[Tensor] = []

        for start in indices.tolist():
            xList.append(source[start : start + cfg.blockSize])
            yList.append(source[start + 1 : start + 1 + cfg.blockSize])

        batchX = torch.stack(xList).to(cfg.device)
        batchY = torch.stack(yList).to(cfg.device)
        return batchX, batchY
