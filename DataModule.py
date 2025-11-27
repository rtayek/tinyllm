# DataModule.py

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

        start_indices: List[int] = [int(v) for v in indices.view(-1).tolist()]  # type: ignore[reportUnknownMemberType]
        for start_idx in start_indices:
            xList.append(source[start_idx : start_idx + modelCfg.blockSize])
            yList.append(source[start_idx + 1 : start_idx + 1 + modelCfg.blockSize])

        batchX = torch.stack(xList).to(trainCfg.device)
        batchY = torch.stack(yList).to(trainCfg.device)
        return batchX, batchY
