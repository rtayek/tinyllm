from __future__ import annotations


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .Config import ModelConfig
from .Transformer import Block


class TinyGpt(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.cfg = cfg

        self.tokenEmbedding = nn.Embedding(cfg.vocabSize, cfg.nEmbed)
        self.positionEmbedding = nn.Embedding(cfg.blockSize, cfg.nEmbed)

        self.blocks = nn.ModuleList([Block(cfg.nEmbed, cfg.nHead, cfg.dropout, cfg.blockSize) for _ in range(cfg.nLayer)])

        self.finalLayerNorm = nn.LayerNorm(cfg.nEmbed)
        self.outputHead = nn.Linear(cfg.nEmbed, cfg.vocabSize, bias=False)

        self.apply(self.initWeights)

    def initWeights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        indices: Tensor,
        targets: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List[Tuple[Tensor, Tensor]]]]:
        if indices.dim() != 2:
            raise ValueError(f"indices must be 2D (batch, time), got {indices.shape}")

        _batch, time = indices.shape
        if time > self.cfg.blockSize:
            raise ValueError(f"Sequence length {time} exceeds blockSize {self.cfg.blockSize}")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError(f"past_key_values length {len(past_key_values)} does not match number of blocks {len(self.blocks)}")

        if indices.dtype != torch.long:
            indices = indices.long()

        past_length = 0
        if past_key_values is not None and past_key_values:
            past_length = past_key_values[0][0].size(2)
        total_length = past_length + time
        if total_length > self.cfg.blockSize:
            if past_key_values is None:
                raise ValueError(f"Sequence length {total_length} exceeds blockSize {self.cfg.blockSize}")
            overflow = total_length - self.cfg.blockSize
            trimmed: List[Tuple[Tensor, Tensor]] = []
            for past_k, past_v in past_key_values:
                trimmed.append((past_k[:, :, overflow:, :], past_v[:, :, overflow:, :]))
            past_key_values = trimmed
            past_length = trimmed[0][0].size(2) if trimmed else 0
            total_length = past_length + time

        tokenEmb = self.tokenEmbedding(indices)
        posIndex = torch.arange(past_length, past_length + time, device=indices.device)
        posEmb = self.positionEmbedding(posIndex).unsqueeze(0)

        x = tokenEmb + posEmb
        present_key_values: List[Tuple[Tensor, Tensor]] = []
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, present = block(x, past_key_value=past)
            if use_cache:
                present_key_values.append(present)
        x = self.finalLayerNorm(x)
        logits = self.outputHead(x)

        if targets is None:
            return logits, None, present_key_values if use_cache else None

        if targets.shape != indices.shape:
            raise ValueError(f"targets must have same shape as indices; got {targets.shape} vs {indices.shape}")

        if targets.dtype != torch.long:
            targets = targets.long()

        loss = F.cross_entropy(
            logits.view(-1, self.cfg.vocabSize),
            targets.view(-1),
        )
        return logits, loss, present_key_values if use_cache else None

    def generate(self, indices: Tensor, maxNewTokens: int) -> Tensor:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None
            for _ in range(maxNewTokens):
                input_indices = indices[:, -self.cfg.blockSize :] if past_key_values is None else indices[:, -1:]
                logits, _, past_key_values = self(input_indices, past_key_values=past_key_values, use_cache=True)
                logitsLast = logits[:, -1, :]
                probs = F.softmax(logitsLast, dim=-1)
                nextToken = torch.multinomial(probs, num_samples=1)
                indices = torch.cat((indices, nextToken), dim=1)
        if was_training:
            self.train()
        return indices
