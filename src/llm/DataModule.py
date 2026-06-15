from __future__ import annotations

from typing import Protocol, Sequence
import logging
import torch

from .Config import ModelConfig, TrainConfig


class Tokenizer(Protocol):
    vocabSize: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...


class Utf8ByteTokenizer:
    """Minimal tokenizer that maps UTF-8 bytes to token IDs."""

    vocabSize: int = 256

    def encode(self, text: str) -> list[int]:
        return [int(b) for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

class SequenceDataModule:
    def __init__(
        self,
        modelConfig: ModelConfig,
        trainConfig: TrainConfig,
        sequence: torch.Tensor,
        validationSequence: torch.Tensor | None = None,
        testSequence: torch.Tensor | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.modelConfig = modelConfig
        self.trainConfig = trainConfig
        self.logger = logger or logging.getLogger(__name__)

        if validationSequence is None:
            splitIndex = int(0.9 * sequence.size(0))
            self.trainSequence = sequence[:splitIndex]
            self.valSequence = sequence[splitIndex:]
        else:
            self.trainSequence = sequence
            self.valSequence = validationSequence
        self.testSequence = testSequence

        self.logger.info(
            "Loaded sequence dataset: train=%d, val=%d, test=%s",
            self.trainSequence.size(0),
            self.valSequence.size(0),
            self.testSequence.size(0) if self.testSequence is not None else "none",
        )

    def _getSource(self, split: str) -> torch.Tensor:
        if split == "train":
            return self.trainSequence
        if split == "val":
            return self.valSequence
        if split == "test" and self.testSequence is not None:
            return self.testSequence
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
            raise ValueError("getBatch requires an explicit generator")

        high = source.size(0) - modelConfig.blockSize
        indices = torch.randint(
            low=0,
            high=high,
            size=(trainConfig.batchSize,),
            generator=generator,
        )

        offsets = torch.arange(modelConfig.blockSize)
        positions = indices.unsqueeze(1) + offsets.unsqueeze(0)
        batchX = source[positions].to(trainConfig.device)
        batchY = source[positions + 1].to(trainConfig.device)
        return batchX, batchY


class ByteDataModule(SequenceDataModule):
    @staticmethod
    def _read_bytes(path: str) -> torch.Tensor:
        with open(path, "rb") as f:
            return torch.tensor(list(f.read()), dtype=torch.long)

    def __init__(
        self,
        modelConfig: ModelConfig,
        trainConfig: TrainConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        sequence = self._read_bytes(trainConfig.dataPath)
        validationSequence = (
            self._read_bytes(trainConfig.validationDataPath)
            if trainConfig.validationDataPath
            else None
        )
        testSequence = (
            self._read_bytes(trainConfig.testDataPath)
            if trainConfig.testDataPath
            else None
        )
        super().__init__(
            modelConfig,
            trainConfig,
            sequence,
            validationSequence,
            testSequence,
            logger,
        )


class TokenDataModule(SequenceDataModule):
    @staticmethod
    def _read_tokens(path: str, tokenizer: Tokenizer) -> torch.Tensor:
        with open(path, "r", encoding="utf-8") as f:
            ids = list(tokenizer.encode(f.read()))
        if not ids:
            raise ValueError(f"Tokenized dataset is empty: {path}")
        return torch.tensor(ids, dtype=torch.long)

    def __init__(
        self,
        modelConfig: ModelConfig,
        trainConfig: TrainConfig,
        tokenizer: Tokenizer,
        logger: logging.Logger | None = None,
    ) -> None:
        sequence = self._read_tokens(trainConfig.dataPath, tokenizer)
        validationSequence = (
            self._read_tokens(trainConfig.validationDataPath, tokenizer)
            if trainConfig.validationDataPath
            else None
        )
        testSequence = (
            self._read_tokens(trainConfig.testDataPath, tokenizer)
            if trainConfig.testDataPath
            else None
        )
        for splitName, splitSequence in (
            ("train", sequence),
            ("validation", validationSequence),
            ("test", testSequence),
        ):
            if splitSequence is not None and int(splitSequence.max()) >= modelConfig.vocabSize:
                raise ValueError(
                    f"Token id {int(splitSequence.max())} in {splitName} split "
                    f"exceeds vocabSize={modelConfig.vocabSize}"
                )

        super().__init__(
            modelConfig,
            trainConfig,
            sequence,
            validationSequence,
            testSequence,
            logger,
        )

        self.tokenizer: Tokenizer = tokenizer
