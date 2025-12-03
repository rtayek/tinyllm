from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadSelfAttention(nn.Module):
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

    def forward(self, x: Tensor, past_key_value: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
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
            # Slide the cache window to keep only the most recent context that fits blockSize.
            offset = total_len - max_len
            k = k[:, :, offset:, :]
            v = v[:, :, offset:, :]
            past_len = k.size(2) - time
            total_len = max_len

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


class Block(nn.Module):
    def __init__(self, nEmbed: int, nHead: int, dropout: float, blockSize: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.selfAttention = MultiHeadSelfAttention(nEmbed, nHead, dropout, blockSize)
        self.layerNorm1 = nn.LayerNorm(nEmbed)
        self.layerNorm2 = nn.LayerNorm(nEmbed)

        self.mlp = nn.Sequential(
            nn.Linear(nEmbed, 4 * nEmbed),
            nn.GELU(),
            nn.Linear(4 * nEmbed, nEmbed),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, past_key_value: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        att_out, present = self.selfAttention(self.layerNorm1(x), past_key_value=past_key_value)
        x = x + att_out
        x = x + self.mlp(self.layerNorm2(x))
        return x, present
