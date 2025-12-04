from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
import logging

import torch

from .tensor_utils import tensor_to_int_list

if TYPE_CHECKING:
    from .Model import TinyGPTLanguageModel

class AutoregressiveGenerator:
    def __init__(self, model: "TinyGPTLanguageModel", device: str, logger: Optional[logging.Logger] = None) -> None:
        self.model: "TinyGPTLanguageModel" = model
        self.device: str = device
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    def generateBytes(self, maxNewTokens: int = 200, prompt: str = "") -> bytes:
        if prompt:
            promptBytes = prompt.encode("utf-8")
            promptTensor = torch.tensor(list(promptBytes), dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            promptTensor = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        with torch.no_grad():
            generated: torch.Tensor = self.model.generate_autoregressive(promptTensor, maxNewTokens=maxNewTokens)

        firstSeq: torch.Tensor = generated[0]
        raw_list: List[int] = tensor_to_int_list(
            firstSeq.to(dtype=torch.long).view(-1)
        )
        return bytes(raw_list)

    def generateText(self, maxNewTokens: int = 200, errors: str = "ignore", prompt: str = "") -> str:
        data = self.generateBytes(maxNewTokens=maxNewTokens, prompt=prompt)
        return data.decode("utf-8", errors=errors)

    def log_sample(self, maxNewTokens: int = 200, prompt: str = "") -> None:
        text = self.generateText(maxNewTokens=maxNewTokens, prompt=prompt)
        self.logger.info("Sampled text:")
        self.logger.info(text)
