# Config.py

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ModelConfig:
    # model
    vocabSize: int = 256          # byte-level
    blockSize: int = 128
    nEmbed: int = 256
    nHead: int = 4
    nLayer: int = 4
    dropout: float = 0.2


@dataclass(frozen=True)
class TrainConfig:
    # training
    batchSize: int = 32
    learningRate: float = 5e-5
    warmupFrac: float = 0.1
    maxSteps: int = 5000
    evalInterval: int = 100
    evalIters: int = 100
    weightDecay: float = 0.02
    earlyStopPatience: int = 2      # number of evals
    earlyStopDelta: float = 0.003  # minimum improvement in val loss

    # paths
    ckptPath: str = "checkpoints/tiny_llm.pt"
    dataPath: str = "testData/input.txt"

    # hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
