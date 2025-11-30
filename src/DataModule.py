# DataModule.py

from __future__ import annotations

import os
from typing import Tuple, List, Optional
import logging

import torch
from torch import Tensor

from Config import ModelConfig, TrainConfig
from tensor_utils import tensor_to_int_list


class ByteDataModule:
    def __init__(self, modelConfig: ModelConfig, trainConfig: TrainConfig, logger: Optional[logging.Logger] = None) -> None:
        self.modelConfig = modelConfig
        self.trainConfig = trainConfig
        self.logger = logger or logging.getLogger(__name__)
        self.trainData: Tensor
        self.valData: Tensor

        self.loadAndSplit()

    def loadAndSplit(self) -> None:
        self.logger.info("Loading dataset from %s", self.trainConfig.dataPath)
        if not os.path.exists(self.trainConfig.dataPath):
            raise FileNotFoundError(self.trainConfig.dataPath)

        with open(self.trainConfig.dataPath, "rb") as f:
            dataBytes = f.read()

        data = torch.tensor(list(dataBytes), dtype=torch.long)
        boundary = int(0.9 * len(data))
        self.trainData = data[:boundary]
        self.valData = data[boundary:]
        self.logger.info(
            "Loaded %d bytes (%d train, %d val) from %s",
            len(data),
            self.trainData.size(0),
            self.valData.size(0),
            self.trainConfig.dataPath,
        )

    def getBatch(self, split: str, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
        if split == "train":
            source = self.trainData
        elif split == "val":
            source = self.valData
        else:
            raise ValueError(f"Unknown split: {split}")

        modelConfig = self.modelConfig
        trainConfig = self.trainConfig
        min_required = modelConfig.blockSize + 2  # need at least one start index
        if len(source) < min_required:
            raise ValueError(
                f"Dataset split '{split}' is too small for blockSize={modelConfig.blockSize}; "
                f"need at least {min_required} bytes, got {len(source)}."
            )

        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(1337)

        indices = torch.randint(
            low=0,
            high=len(source) - modelConfig.blockSize - 1,
            size=(trainConfig.batchSize,),
            generator=generator,
        )

        xList: List[Tensor] = []
        yList: List[Tensor] = []

        startIndices: List[int] = tensor_to_int_list(indices)
        for startIndex in startIndices:
            xList.append(source[startIndex : startIndex + modelConfig.blockSize])
            yList.append(source[startIndex + 1 : startIndex + 1 + modelConfig.blockSize])

        batchX = torch.stack(xList).to(trainConfig.device)
        batchY = torch.stack(yList).to(trainConfig.device)
        assert batchX.shape == (trainConfig.batchSize, modelConfig.blockSize)
        assert batchY.shape == (trainConfig.batchSize, modelConfig.blockSize)
        return batchX, batchY
