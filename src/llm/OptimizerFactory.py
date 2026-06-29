"""Factories for the optimizer and LR scheduler used by ``LMTrainer``.

Pulling optimizer/scheduler construction behind small factory protocols lets a
caller hand the trainer a *fresh* optimizer per use without the trainer knowing
how it is built. This matters for iterated training (e.g. expert-iteration
self-improvement), where each round wants a clean optimizer/scheduler over the
same model.

The defaults reproduce the trainer's previous inline construction exactly:
AdamW with the config's learning rate / weight decay, and a warmup-cosine
schedule over ``maxSteps`` with ``warmupFrac`` warmup.
"""
from __future__ import annotations

from typing import Protocol

import torch

from .Config import TrainConfig
from .LRScheduleStrategy import WarmupCosineStrategy
from .Model import TinyGPTLanguageModel


class OptimizerFactory(Protocol):
    """Builds a fresh optimizer for a model under a given training config."""

    def __call__(
        self,
        model: TinyGPTLanguageModel,
        trainConfig: TrainConfig,
    ) -> torch.optim.Optimizer: ...


class SchedulerFactory(Protocol):
    """Builds a fresh LR strategy wrapping a given optimizer."""

    def __call__(
        self,
        optimizer: torch.optim.Optimizer,
        trainConfig: TrainConfig,
    ) -> WarmupCosineStrategy: ...


def default_optimizer_factory(
    model: TinyGPTLanguageModel,
    trainConfig: TrainConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=trainConfig.learningRate,
        weight_decay=trainConfig.weightDecay,
    )


def default_scheduler_factory(
    optimizer: torch.optim.Optimizer,
    trainConfig: TrainConfig,
) -> WarmupCosineStrategy:
    return WarmupCosineStrategy(
        optimizer,
        max_steps=trainConfig.maxSteps,
        warmup_frac=trainConfig.warmupFrac,
    )
