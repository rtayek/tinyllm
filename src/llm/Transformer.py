from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .Config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, nEmbed: int, nHead: int, dropout: float, blockSize: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        if nEmbed % nHead != 0:
            raise ValueError("nEmbed must be divisible by nHead")
        self.nHead = nHead
        self.headDim = nEmbed // nHead

        self.key = nn.Linear(nEmbed, nEmbed, bias=False)
        self.query = nn.Linear(nEmbed, nEmbed, bias=False)
        self.value = nn.Linear(nEmbed, nEmbed, bias=False)

        self.proj = nn.Linear(nEmbed, nEmbed)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(blockSize, blockSize))
        self.register_buffer("mask", mask.view(1, 1, blockSize, blockSize))

    def forward(self, x: Tensor, past_key_value: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, time, channels = x.shape

        k_new = self.key(x).view(batch, time, self.nHead, self.headDim).transpose(1, 2)
        q = self.query(x).view(batch, time, self.nHead, self.headDim).transpose(1, 2)
        v_new = self.value(x).view(batch, time, self.nHead, self.headDim).transpose(1, 2)

        if past_key_value is None:
            past_len = 0
            k = k_new
            v = v_new
        else:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
            k = torch.cat([past_k, k_new], dim=2)
            v = torch.cat([past_v, v_new], dim=2)

        total_len = past_len + time
        max_len = self.mask.size(-1)
        if total_len > max_len:
            raise ValueError(
                f"Cached sequence length {total_len} exceeds blockSize {max_len}"
            )

        mask = self.mask[:, :, total_len - time : total_len, :total_len]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.headDim))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, channels)
        out = self.proj(out)
        out = self.dropout(out)
        return out, (k, v)


class DecoderBlock(nn.Module):
    def __init__(self, nEmbed: int, nHead: int, dropout: float, blockSize: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.selfAttention = CausalSelfAttention(nEmbed, nHead, dropout, blockSize)
        self.layerNorm1 = nn.LayerNorm(nEmbed)
        self.layerNorm2 = nn.LayerNorm(nEmbed)

        self.mlp = nn.Sequential( # multi‑layer perceptron
            nn.Linear(nEmbed, 4 * nEmbed),
            nn.GELU(),
            nn.Linear(4 * nEmbed, nEmbed),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, past_key_value: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        att_out, present = self.selfAttention(self.layerNorm1(x), past_key_value=past_key_value)
        x = x + att_out
        x = x + self.mlp(self.layerNorm2(x))
        return x, present

class DecoderCore(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.cfg = cfg
        self.tokenEmbedding = nn.Embedding(cfg.vocabSize, cfg.nEmbed)
        self.positionEmbedding = nn.Embedding(cfg.blockSize, cfg.nEmbed)
        self.blocks = nn.ModuleList(
            [DecoderBlock(cfg.nEmbed, cfg.nHead, cfg.dropout, cfg.blockSize) for _ in range(cfg.nLayer)]
        )
        self.finalLayerNorm = nn.LayerNorm(cfg.nEmbed)

    def forward(
        self,
        indices: Tensor,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]] | None]:
        if indices.dim() != 2:
            raise ValueError(f"indices must be 2D (batch, time), got {indices.shape}")

        _batch_size, time = indices.shape

        if time > self.cfg.blockSize:
            raise ValueError(f"Sequence length {time} exceeds blockSize {self.cfg.blockSize}")

        if indices.dtype != torch.long:
            indices = indices.long()

        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError(
                f"past_key_values length {len(past_key_values)} does not match number of blocks {len(self.blocks)}"
            )

        past_len = 0
        if past_key_values is not None:
            past_lengths = {past_k.size(2) for past_k, _ in past_key_values}
            if len(past_lengths) != 1:
                raise ValueError("All cached layers must have the same sequence length")
            past_len = past_lengths.pop()

        if past_len + time > self.cfg.blockSize:
            raise ValueError(
                f"Cached sequence length {past_len + time} exceeds blockSize "
                f"{self.cfg.blockSize}"
            )

        # Cached tokens retain their original learned absolute positions.
        device = indices.device
        positions = torch.arange(
            past_len,
            past_len + time,
            device=device,
        ).unsqueeze(0)

        tok_emb = self.tokenEmbedding(indices)      # (B, T, nEmbed)
        pos_emb = self.positionEmbedding(positions) # (1, T, nEmbed)
        x = tok_emb + pos_emb

        new_kv: list[tuple[Tensor, Tensor]] | None = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past = None
            if past_key_values is not None:
                past = past_key_values[i]

            x, present = block(x, past_key_value=past)

            if use_cache and new_kv is not None:
                new_kv.append(present)

        x = self.finalLayerNorm(x)

        return x, new_kv
