"""Tiny LLM package."""

from .Config import RunConfig, ModelConfig, TrainConfig, RunPaths
from .Model import TinyGPTLanguageModel
from .Trainer import LMTrainer
from .OptimizerFactory import (
    OptimizerFactory,
    SchedulerFactory,
    default_optimizer_factory,
    default_scheduler_factory,
)
from .DataModule import ByteDataModule, DataModuleConfig, TokenDataModule, SequenceDataModule
from .EvalResult import EvalResult
from .EvaluationProbe import EvalContext, EvaluationProbe
from .GeneratedCandidate import GeneratedCandidate
from .EvaluationMode import (
    BaselineEvaluator,
    CorruptionEvaluator,
    FullSplitEvaluator,
    PerBookEvaluator,
    SampledLossEvaluator,
)
from .TextGenerator import AutoregressiveGenerator
from .TrainingCallback import TrainingCallback, LoggingCallback, MetricsCallback, CheckpointCallback, TrainingCurveCallback

__all__ = [
    "RunConfig",
    "ModelConfig",
    "TrainConfig",
    "RunPaths",
    "TinyGPTLanguageModel",
    "LMTrainer",
    "OptimizerFactory",
    "SchedulerFactory",
    "default_optimizer_factory",
    "default_scheduler_factory",
    "ByteDataModule",
    "DataModuleConfig",
    "TokenDataModule",
    "SequenceDataModule",
    "EvalResult",
    "EvalContext",
    "EvaluationProbe",
    "GeneratedCandidate",
    "BaselineEvaluator",
    "CorruptionEvaluator",
    "FullSplitEvaluator",
    "PerBookEvaluator",
    "SampledLossEvaluator",
    "AutoregressiveGenerator",
    "TrainingCallback",
    "LoggingCallback",
    "MetricsCallback",
    "CheckpointCallback",
    "TrainingCurveCallback",
]
