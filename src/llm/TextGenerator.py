from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import logging

import torch

from .tensor_utils import tensor_to_int_list

if TYPE_CHECKING:
    from .Model import TinyGPTLanguageModel

class AutoregressiveGenerator:
    def __init__(self, model: "TinyGPTLanguageModel", device: str, logger: logging.Logger | None = None) -> None:
        self.model: "TinyGPTLanguageModel" = model
        self.device: str = device
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    def generateBytes(
        self,
        maxNewTokens: int = 200,
        prompt: str = "",
        temperature: float = 1.0,
        topK: int | None = None,
        seed: int | None = None,
    ) -> bytes:
        if prompt:
            promptBytes = prompt.encode("utf-8")
            promptTensor = torch.tensor(list(promptBytes), dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            promptTensor = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        with torch.no_grad():
            generated: torch.Tensor = self.model.generate_autoregressive(
                promptTensor,
                maxNewTokens=maxNewTokens,
                temperature=temperature,
                topK=topK,
                seed=seed,
            )

        firstSeq: torch.Tensor = generated[0]
        raw_list: list[int] = tensor_to_int_list(
            firstSeq.to(dtype=torch.long).view(-1)
        )
        return bytes(raw_list)

    def generateText(
        self,
        maxNewTokens: int = 200,
        errors: str = "replace",
        prompt: str = "",
        temperature: float = 1.0,
        topK: int | None = None,
        seed: int | None = None,
    ) -> str:
        data = self.generateBytes(
            maxNewTokens=maxNewTokens,
            prompt=prompt,
            temperature=temperature,
            topK=topK,
            seed=seed,
        )
        return data.decode("utf-8", errors=errors)

    def logSample(self, maxNewTokens: int = 200, prompt: str = "") -> None:
        text = self.generateText(maxNewTokens=maxNewTokens, prompt=prompt)
        self.logger.info("Sampled text:")
        self.logger.info(text)

    def saveSample(
        self,
        maxNewTokens: int = 200,
        prompt: str = "",
        path: str | Path = Path("tmp") / "sample.txt",
    ) -> None:
        text = self.generateText(maxNewTokens=maxNewTokens, prompt=prompt)
        outputPath = Path(path)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        with outputPath.open("w", encoding="utf-8") as f:
            f.write(text)
        self.logger.info("Sampled text saved to %s", outputPath)
