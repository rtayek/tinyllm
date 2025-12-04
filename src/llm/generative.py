# src/llm/generative.py

from __future__ import annotations

from typing import Protocol, Optional, runtime_checkable
import logging

from .Model import TinyGPTLanguageModel
from .Config import TrainConfig
from .TextGenerator import AutoregressiveGenerator


@runtime_checkable
class GenerativeLM(Protocol):
    """
    Minimal interface for a text-generating language model.

    Baby-step version:
      - one method
      - prompt in, text out
    """

    def generate_text(self, prompt: Optional[str] = None, max_new_tokens: int = 128) -> str:
        ...


class TinyGptGenerative:
    """
    Adapter that wraps TinyGpt + TextGenerator behind the GenerativeLM interface.

    This does NOT change TinyGpt or TextGenerator.
    It just gives you a clean, narrow interface for higher-level code
    (future RAG / agents) to depend on.
    """

    def __init__(self, model: TinyGPTLanguageModel, train_cfg: TrainConfig, logger: logging.Logger) -> None:
        self._model: TinyGPTLanguageModel = model
        self._train_cfg: TrainConfig = train_cfg
        self._logger: logging.Logger = logger

        # Reuse your existing TextGenerator, which already knows how
        # to turn the model into human-readable text.
        self._text_gen: AutoregressiveGenerator = AutoregressiveGenerator(model, train_cfg.device, logger)

    def generate_text(self, prompt: Optional[str] = None, max_new_tokens: int = 128) -> str:
        prompt_text = prompt or ""
        return self._text_gen.generateText(maxNewTokens=max_new_tokens, prompt=prompt_text)
