# Model.py

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from Config import ModelConfig


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

    def forward(self, x: Tensor) -> Tensor:
        batch, time, channels = x.shape

        k = self.key(x).view(batch, time, self.nHead, self.headDim).transpose(1, 2)
        q = self.query(x).view(batch, time, self.nHead, self.headDim).transpose(1, 2)
        v = self.value(x).view(batch, time, self.nHead, self.headDim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.headDim))
        att = att.masked_fill(self.mask[:, :, :time, :time] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, channels)
        out = self.proj(out)
        out = self.dropout(out)
        return out


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

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.selfAttention(self.layerNorm1(x))
        x = x + self.mlp(self.layerNorm2(x))
        return x


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

    def forward(self, indices: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if indices.dim() != 2:
            raise ValueError(f"indices must be 2D (batch, time), got {indices.shape}")

        _batch, time = indices.shape
        if time > self.cfg.blockSize:
            raise ValueError(f"Sequence length {time} exceeds blockSize {self.cfg.blockSize}")

        if indices.dtype != torch.long:
            indices = indices.long()

        tokenEmb = self.tokenEmbedding(indices)
        posIndex = torch.arange(time, device=indices.device)
        posEmb = self.positionEmbedding(posIndex).unsqueeze(0)

        x = tokenEmb + posEmb
        for block in self.blocks:
            x = block(x)
        x = self.finalLayerNorm(x)
        logits = self.outputHead(x)

        if targets is None:
            return logits, None

        if targets.shape != indices.shape:
            raise ValueError(f"targets must have same shape as indices; got {targets.shape} vs {indices.shape}")

        if targets.dtype != torch.long:
            targets = targets.long()

        loss = F.cross_entropy(
            logits.view(-1, self.cfg.vocabSize),
            targets.view(-1),
        )
        return logits, loss

    def generate(self, indices: Tensor, maxNewTokens: int) -> Tensor:
        self.eval()
        with torch.no_grad():
            for _ in range(maxNewTokens):
                indicesCond = indices[:, -self.cfg.blockSize:]
                logits, _ = self(indicesCond)
                logitsLast = logits[:, -1, :]
                probs = F.softmax(logitsLast, dim=-1)
                nextToken = torch.multinomial(probs, num_samples=1)
                indices = torch.cat((indices, nextToken), dim=1)
        self.train()
        return indices
