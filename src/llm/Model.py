from __future__ import annotations


import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .Config import ModelConfig
from .Transformer import DecoderCore


logger = logging.getLogger(__name__)


class TinyGPTLanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.cfg = cfg
        self.core = DecoderCore(cfg)
        self.lmHead = nn.Linear(cfg.nEmbed, cfg.vocabSize, bias=False)
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
        targets: Tensor | None = None,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None, list[tuple[Tensor, Tensor]] | None]:
        hidden, new_kv = self.core(
            indices,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lmHead(hidden)

        loss: Tensor | None = None
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
    def generate_autoregressive(
        self,
        indices: Tensor,
        maxNewTokens: int,
        temperature: float = 1.0,
        topK: int | None = None,
        seed: int | None = None,
    ) -> Tensor:
        if indices.dim() != 2:
            raise ValueError(f"indices must be 2D (batch, time), got {indices.shape}")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if topK is not None and topK <= 0:
            raise ValueError("topK must be greater than zero")
        if seed is not None and seed < 0:
            raise ValueError("seed must be non-negative")

        sample_generator: torch.Generator | None = None
        if seed is not None:
            sample_generator = torch.Generator(device=indices.device)
            sample_generator.manual_seed(seed)

        was_training = self.training
        self.eval()
        try:
            past_key_values: list[tuple[Tensor, Tensor]] | None = None
            for _ in range(maxNewTokens):
                if self.cfg.use_cache:
                    cache_is_full = (
                        past_key_values is not None
                        and past_key_values[0][0].size(2) >= self.cfg.blockSize
                    )
                    if past_key_values is None or cache_is_full:
                        # Rebuild at the context boundary because learned
                        # absolute positions cannot be shifted in cached keys.
                        if cache_is_full:
                            logger.debug(
                                "KV cache reached blockSize=%d; rebuilding "
                                "from the current context window",
                                self.cfg.blockSize,
                            )
                        past_key_values = None
                        input_indices = indices[:, -self.cfg.blockSize :]
                    else:
                        input_indices = indices[:, -1:]
                else:
                    input_indices = indices[:, -self.cfg.blockSize :]

                logits, _, new_past_key_values = self(
                    input_indices,
                    past_key_values=past_key_values if self.cfg.use_cache else None,
                    use_cache=self.cfg.use_cache,
                )

                if self.cfg.use_cache:
                    past_key_values = new_past_key_values

                logitsLast = logits[:, -1, :] / temperature
                if topK is not None:
                    effective_top_k = min(topK, logitsLast.size(-1))
                    top_logits, top_indices = torch.topk(
                        logitsLast,
                        effective_top_k,
                        dim=-1,
                    )
                    probs = F.softmax(top_logits, dim=-1)
                    sampled_index = torch.multinomial(
                        probs,
                        num_samples=1,
                        generator=sample_generator,
                    )
                    nextToken = torch.gather(top_indices, -1, sampled_index)
                else:
                    probs = F.softmax(logitsLast, dim=-1)
                    nextToken = torch.multinomial(
                        probs,
                        num_samples=1,
                        generator=sample_generator,
                    )
                indices = torch.cat((indices, nextToken), dim=1)
        finally:
            if was_training:
                self.train()

        return indices
