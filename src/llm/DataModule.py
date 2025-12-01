from __future__ import annotations

from typing import Sequence
import logging
import torch

from .Config import ModelConfig, TrainConfig
from .tensor_utils import tensor_to_int_list


class Utf8ByteTokenizer:
    """Minimal tokenizer that maps UTF-8 bytes to token IDs."""

    vocabSize: int = 256

    def encode(self, text: str) -> list[int]:
        return [int(b) for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="ignore")

class SequenceDataModule:
    def __init__(
        self,
        modelConfig: ModelConfig,
        trainConfig: TrainConfig,
        sequence: torch.Tensor,
        logger: logging.Logger | None = None,
    ) -> None:
        self.modelConfig = modelConfig
        self.trainConfig = trainConfig
        self.logger = logger or logging.getLogger(__name__)

        splitIndex = int(0.9 * sequence.size(0))
        self.trainSequence = sequence[:splitIndex]
        self.valSequence = sequence[splitIndex:]

        self.logger.info(
            "Loaded sequence dataset: total=%d, train=%d, val=%d",
            sequence.size(0),
            self.trainSequence.size(0),
            self.valSequence.size(0),
        )

    def _getSource(self, split: str) -> torch.Tensor:
        if split == "train":
            return self.trainSequence
        if split == "val":
            return self.valSequence
        raise ValueError(f"Unknown split: {split}")

    def getBatch(
        self,
        split: str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modelConfig = self.modelConfig
        trainConfig = self.trainConfig
        source = self._getSource(split)

        minRequired = modelConfig.blockSize + 1
        if source.size(0) < minRequired:
            raise ValueError(
                f"Dataset split '{split}' too small for blockSize {modelConfig.blockSize}"
            )

        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(1337)

        high = source.size(0) - modelConfig.blockSize - 1
        indices = torch.randint(
            low=0,
            high=high,
            size=(trainConfig.batchSize,),
            generator=generator,
        )

        xList: list[torch.Tensor] = []
        yList: list[torch.Tensor] = []

        start_indices = tensor_to_int_list(indices)
        for start in start_indices:
            xList.append(
                source[start : start + modelConfig.blockSize]
            )
            yList.append(
                source[start + 1 : start + 1 + modelConfig.blockSize]
            )

        batchX = torch.stack(xList).to(trainConfig.device)
        batchY = torch.stack(yList).to(trainConfig.device)
        return batchX, batchY


class ByteDataModule(SequenceDataModule):
    def __init__(
        self,
        modelConfig: ModelConfig,
        trainConfig: TrainConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        with open(trainConfig.dataPath, "rb") as f:
            data = f.read()
        sequence = torch.tensor(list(data), dtype=torch.long)
        super().__init__(modelConfig, trainConfig, sequence, logger)


class TokenDataModule(SequenceDataModule):
    def __init__(
        self,
        modelConfig: ModelConfig,
        trainConfig: TrainConfig,
        tokenizer: Utf8ByteTokenizer,
        logger: logging.Logger | None = None,
    ) -> None:
        with open(trainConfig.dataPath, "r", encoding="utf-8") as f:
            text = f.read()

        ids = list(tokenizer.encode(text))
        if not ids:
            raise ValueError("Tokenized dataset is empty")

        maxId = max(ids)
        if maxId >= modelConfig.vocabSize:
            raise ValueError(
                f"Token id {maxId} exceeds vocabSize={modelConfig.vocabSize}"
            )

        sequence = torch.tensor(ids, dtype=torch.long)
        super().__init__(modelConfig, trainConfig, sequence, logger)

        self.tokenizer: Utf8ByteTokenizer = tokenizer
