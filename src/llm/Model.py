from __future__ import annotations


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .Config import ModelConfig
from .Transformer import DecoderCore


class TinyGPTLanguageModel(nn.Module):
    """
    Language model built on top of TransformerCore.

    Responsibilities:
      - hold a TransformerCore
      - add an lm_head to map hidden states → vocab logits
      - compute cross-entropy LM loss when targets are provided
      - implement autoregressive generate(...)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.cfg = cfg

        # The pure transformer stack
        self.core = DecoderCore(cfg)

        # LM head: projects hidden states to vocab logits
        self.lmHead = nn.Linear(cfg.nEmbed, cfg.vocabSize, bias=False)

        # Weight init for all submodules (core + head)
        self.apply(self.initWeights)

    def initWeights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        indices: Tensor,
        targets: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List[Tuple[Tensor, Tensor]]]]:
        """
        Args:
            indices: (batch, time) token IDs
            targets: optional (batch, time) for LM loss
            past_key_values: optional kv-cache for autoregressive decoding
            use_cache: if True, returns updated kv-cache

        Returns:
            logits: (batch, time, vocabSize)
            loss: scalar tensor or None
            new_kv: list[(k, v)] or None
        """
        # Let the core do the transformer work
        hidden, new_kv = self.core(
            indices,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Project hidden states to vocab logits
        logits = self.lmHead(hidden)

        loss: Optional[Tensor] = None
        if targets is not None:
            if targets.shape != indices.shape:
                raise ValueError(
                    f"targets shape {targets.shape} must match indices shape {indices.shape}"
                )
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, new_kv

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def generate_autoregressive(self, indices: Tensor, maxNewTokens: int) -> Tensor:
        """
        Autoregressive generation:

        Repeatedly:
          - run the model on the current context (with kv-cache)
          - sample the next token from logits
          - append to the sequence
        """
        if indices.dim() != 2:
            raise ValueError(f"indices must be 2D (batch, time), got {indices.shape}")

        was_training = self.training
        self.eval()
        try:
            past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None
            for _ in range(maxNewTokens):
                if self.cfg.use_cache:
                    # If we have no cache yet, feed last blockSize tokens.
                    # Once cache exists, we can feed just the last token.
                    if past_key_values is None:
                        input_indices = indices[:, -self.cfg.blockSize :]
                    else:
                        input_indices = indices[:, -1:]
                else:
                    # If caching is disabled, always feed the full context (up to blockSize)
                    input_indices = indices[:, -self.cfg.blockSize :]

                logits, _, new_past_key_values = self(
                    input_indices,
                    past_key_values=past_key_values if self.cfg.use_cache else None,
                    use_cache=self.cfg.use_cache,
                )

                if self.cfg.use_cache:
                    past_key_values = new_past_key_values

                logitsLast = logits[:, -1, :]          # (B, vocab)
                probs = F.softmax(logitsLast, dim=-1)
                nextToken = torch.multinomial(probs, num_samples=1)  # (B, 1)
                indices = torch.cat((indices, nextToken), dim=1)
        finally:
            if was_training:
                self.train()

        return indices
