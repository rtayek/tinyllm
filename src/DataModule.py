# DataModule.py

from __future__ import annotations

import os
from typing import Tuple, List, Optional

import torch
from torch import Tensor

from Config import ModelConfig, TrainConfig
from tensor_utils import tensor_to_int_list


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

        startIndices: List[int] = tensor_to_int_list(indices)
        for startIndex in startIndices:
            xList.append(source[startIndex : startIndex + modelCfg.blockSize])
            yList.append(source[startIndex + 1 : startIndex + 1 + modelCfg.blockSize])

        batchX = torch.stack(xList).to(trainCfg.device)
        batchY = torch.stack(yList).to(trainCfg.device)
        assert batchX.shape == (trainCfg.batchSize, modelCfg.blockSize)
        assert batchY.shape == (trainCfg.batchSize, modelCfg.blockSize)
        return batchX, batchY
