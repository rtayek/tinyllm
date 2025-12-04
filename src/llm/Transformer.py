from __future__ import annotations

import math
from typing import List, Optional, Tuple

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


class DecoderBlock(nn.Module):
    def __init__(self, nEmbed: int, nHead: int, dropout: float, blockSize: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.selfAttention = CausalSelfAttention(nEmbed, nHead, dropout, blockSize)
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

class DecoderCore(nn.Module):
    """
    Pure Transformer decoder stack:
      - token + position embeddings
      - N Block layers (each with MultiHeadSelfAttention + MLP)
      - final layer norm

    No lm_head, no loss, no text generation. It just maps
    token indices → hidden states (and optional kv-cache).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.cfg = cfg

        # Embeddings
        self.tokenEmbedding = nn.Embedding(cfg.vocabSize, cfg.nEmbed)
        self.positionEmbedding = nn.Embedding(cfg.blockSize, cfg.nEmbed)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList(
            [DecoderBlock(cfg.nEmbed, cfg.nHead, cfg.dropout, cfg.blockSize) for _ in range(cfg.nLayer)]
        )

        # Final layer norm
        self.finalLayerNorm = nn.LayerNorm(cfg.nEmbed)

    def forward(
        self,
        indices: Tensor,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tuple[Tensor, Tensor]]]]:
        """
        Args:
            indices: (batch, time) token IDs
            past_key_values: optional list of (k, v) for each block
            use_cache: if True, returns new kv-cache for autoregressive generation

        Returns:
            hidden_states: (batch, time, nEmbed)
            new_kv: list[(k, v)] or None
        """
        if indices.dim() != 2:
            raise ValueError(f"indices must be 2D (batch, time), got {indices.shape}")

        _batch_size, time = indices.shape

        if time > self.cfg.blockSize:
            raise ValueError(f"Sequence length {time} exceeds blockSize {self.cfg.blockSize}")

        if indices.dtype != torch.long:
            indices = indices.long()

        # Validate past_key_values length if provided
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError(
                f"past_key_values length {len(past_key_values)} does not match number of blocks {len(self.blocks)}"
            )

        # Compute position embeddings
        device = indices.device
        positions = torch.arange(0, time, device=device).unsqueeze(0)  # (1, time)

        tok_emb = self.tokenEmbedding(indices)      # (B, T, nEmbed)
        pos_emb = self.positionEmbedding(positions) # (1, T, nEmbed)
        x = tok_emb + pos_emb

        new_kv: Optional[List[Tuple[Tensor, Tensor]]] = [] if use_cache else None

        # Pass through each Transformer block
        for i, block in enumerate(self.blocks):
            past = None
            if past_key_values is not None:
                past = past_key_values[i]

            x, present = block(x, past_key_value=past)

            if use_cache and new_kv is not None:
                new_kv.append(present)

        # Final layer norm
        x = self.finalLayerNorm(x)

        return x, new_kv
