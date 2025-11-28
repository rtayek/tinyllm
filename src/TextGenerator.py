# TextGenerator.py
from __future__ import annotations

from typing import List, Optional
import logging

import torch

from Model import TinyGpt
from tensor_utils import tensor_to_int_list


class TextGenerator:
    def __init__(
        self,
        model: TinyGpt,
        device: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

    def generate_bytes(self, maxNewTokens: int = 200, prompt: str = "") -> bytes:
        if prompt:
            prompt_bytes = prompt.encode("utf-8")
            prompt_tensor = torch.tensor(list(prompt_bytes), dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            prompt_tensor = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        with torch.no_grad():
            generated: torch.Tensor = self.model.generate(
                prompt_tensor,
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
        prompt: str = "",
    ) -> str:
        data = self.generate_bytes(maxNewTokens=maxNewTokens, prompt=prompt)
        return data.decode("utf-8", errors=errors)

    def log_sample(self, maxNewTokens: int = 200, prompt: str = "") -> None:
        text = self.generate_text(maxNewTokens=maxNewTokens, prompt=prompt)
        self.logger.info("Sampled text:")
        self.logger.info(text)
