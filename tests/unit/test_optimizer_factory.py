"""Tests for optimizer/scheduler factory injection into LMTrainer.

The defaults must reproduce the trainer's previous inline construction, and a
caller must be able to inject custom factories (the seam iterated training
needs to build a fresh optimizer per round).
"""
from __future__ import annotations

import torch

from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import DataModuleConfig, SequenceDataModule
from llm.EarlyStopping import EarlyStopping
from llm.Evaluator import Evaluator
from llm.LRScheduleStrategy import WarmupCosineStrategy
from llm.Model import TinyGPTLanguageModel
from llm.OptimizerFactory import (
    default_optimizer_factory,
    default_scheduler_factory,
)
from llm.Trainer import LMTrainer


def _make_trainer(**kwargs: object) -> LMTrainer:
    model_config = ModelConfig(
        vocabSize=16, blockSize=8, nEmbed=16, nHead=2, nLayer=1, dropout=0.0
    )
    train_config = TrainConfig(
        batchSize=2, device="cpu", maxSteps=10, learningRate=1e-3, weightDecay=0.05
    )
    source = torch.arange(64) % 16
    data_module = SequenceDataModule(
        model_config,
        DataModuleConfig.fromTrainConfig(train_config),
        sequence=source,
        validationSequence=source,
    )
    model = TinyGPTLanguageModel(model_config)
    evaluator = Evaluator(
        model, data_module, train_config, EarlyStopping(patience=1, delta=0.0)
    )
    return LMTrainer(
        model_config,
        train_config,
        model,
        data_module,
        evaluator=evaluator,
        **kwargs,  # type: ignore[arg-type]
    )


def test_default_factories_reproduce_previous_construction() -> None:
    trainer = _make_trainer()
    # Default optimizer is AdamW with the config's lr / weight_decay.
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    group = trainer.optimizer.param_groups[0]
    assert group["lr"] == 1e-3
    assert group["weight_decay"] == 0.05
    # Default scheduler is the warmup-cosine strategy.
    assert isinstance(trainer.lrStrategy, WarmupCosineStrategy)


def test_default_optimizer_factory_matches_direct_construction() -> None:
    model_config = ModelConfig(
        vocabSize=16, blockSize=8, nEmbed=16, nHead=2, nLayer=1, dropout=0.0
    )
    train_config = TrainConfig(device="cpu", learningRate=2e-4, weightDecay=0.01)
    model = TinyGPTLanguageModel(model_config)
    optimizer = default_optimizer_factory(model, train_config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 2e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.01

    scheduler = default_scheduler_factory(optimizer, train_config)
    assert isinstance(scheduler, WarmupCosineStrategy)


def test_custom_optimizer_factory_is_used() -> None:
    calls: list[tuple[int, float]] = []

    def sgd_factory(
        model: TinyGPTLanguageModel,
        trainConfig: TrainConfig,
    ) -> torch.optim.Optimizer:
        calls.append((trainConfig.batchSize, trainConfig.learningRate))
        return torch.optim.SGD(model.parameters(), lr=trainConfig.learningRate)

    trainer = _make_trainer(optimizerFactory=sgd_factory)
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    # The factory received the trainer's actual config.
    assert calls == [(2, 1e-3)]


def test_custom_scheduler_factory_receives_optimizer() -> None:
    seen: list[torch.optim.Optimizer] = []

    def scheduler_factory(
        optimizer: torch.optim.Optimizer,
        trainConfig: TrainConfig,
    ) -> WarmupCosineStrategy:
        seen.append(optimizer)
        return WarmupCosineStrategy(
            optimizer, max_steps=trainConfig.maxSteps, warmup_frac=0.0
        )

    trainer = _make_trainer(schedulerFactory=scheduler_factory)
    # The scheduler factory was handed the optimizer the trainer built.
    assert seen == [trainer.optimizer]


def test_fresh_optimizer_per_trainer_for_iterated_training() -> None:
    # Two trainers over the same model config get independent optimizers,
    # which is the property iterated self-improvement relies on.
    a = _make_trainer()
    b = _make_trainer()
    assert a.optimizer is not b.optimizer
