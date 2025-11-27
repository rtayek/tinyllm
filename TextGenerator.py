# TextGenerator.py
from __future__ import annotations

from typing import List, Optional
import logging

import torch

from Config import TrainConfig
from Model import TinyGpt
from tensor_utils import tensor_to_int_list


class TextGenerator:
    def __init__(
        self,
        model: TinyGpt,
        trainCfg: TrainConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.trainCfg = trainCfg
        self.logger = logger or logging.getLogger(__name__)

    def generate_bytes(self, maxNewTokens: int = 200) -> bytes:
        start = torch.zeros((1, 1), dtype=torch.long, device=self.trainCfg.device)

        with torch.no_grad():
            generated: torch.Tensor = self.model.generate(
                start,
                maxNewTokens=maxNewTokens,
            )

        first_seq: torch.Tensor = generated[0]
        raw_list: List[int] = tensor_to_int_list(
            first_seq.to(dtype=torch.long).view(-1)
        )
        return bytes(raw_list)

    def generate_text(
        self,
        maxNewTokens: int = 200,
        errors: str = "ignore",
    ) -> str:
        data = self.generate_bytes(maxNewTokens=maxNewTokens)
        return data.decode("utf-8", errors=errors)

    def log_sample(self, maxNewTokens: int = 200) -> None:
        text = self.generate_text(maxNewTokens=maxNewTokens)
        self.logger.info("Sampled text:")
        self.logger.info(text)
