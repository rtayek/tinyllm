# Config.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    # model
    vocabSize: int = 256          # byte-level
    blockSize: int = 128
    nEmbed: int = 256
    nHead: int = 4
    nLayer: int = 4
    dropout: float = 0.1

    # training
    batchSize: int = 32
    learningRate: float = 3e-4
    maxSteps: int = 5000
    evalInterval: int = 500
    evalIters: int = 100

    # paths
    ckptPath: str = "checkpoints/tiny_llm.pt"
    dataPath: str = "testData/input.txt"

    # hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
