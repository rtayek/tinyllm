# DataModule.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
from typing import Tuple, List, Optional

import torch
from torch import Tensor

from Config import ModelConfig, TrainConfig


class ByteDataModule:
    def __init__(self, modelCfg: ModelConfig, trainCfg: TrainConfig) -> None:
        self.modelCfg = modelCfg
        self.trainCfg = trainCfg
        self.trainData: Tensor
        self.valData: Tensor

        self.loadAndSplit()

    def loadAndSplit(self) -> None:
        if not os.path.exists(self.trainCfg.dataPath):
            raise FileNotFoundError(self.trainCfg.dataPath)

        with open(self.trainCfg.dataPath, "rb") as f:
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

        modelCfg = self.modelCfg
        trainCfg = self.trainCfg

        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(1337)

        indices = torch.randint(
            low=0,
            high=len(source) - modelCfg.blockSize - 1,
            size=(trainCfg.batchSize,),
            generator=generator,
        )

        xList: List[Tensor] = []
        yList: List[Tensor] = []

        for start in indices.tolist():
            xList.append(source[start : start + modelCfg.blockSize])
            yList.append(source[start + 1 : start + 1 + modelCfg.blockSize])

        batchX = torch.stack(xList).to(trainCfg.device)
        batchY = torch.stack(yList).to(trainCfg.device)
        return batchX, batchY
