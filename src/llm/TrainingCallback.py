"""Training event callbacks.

The ``TrainingCallback`` protocol defines two lifecycle hooks that fire during
training.  Concrete callbacks implement one or both hooks and are registered on
``LMTrainer`` at construction time.  The trainer calls them in registration
order, which makes the full event flow visible in one place (``buildTrainer``
in ``train_app.py``).

Hooks
-----
on_eval(result, is_best)
    Called after every evaluation pass.  ``is_best`` is True when ``result``
    produced a new absolute validation-loss minimum.

on_train_end(curve)
    Called once after the training loop exits by exhausting steps or early
    stopping.  ``curve`` is the list of (step, train_loss, val_loss) tuples
    accumulated during the run.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable, Protocol

from .Config import ModelConfig, TrainConfig
from .Evaluator import EvalResult


class TrainingCallback:
    def on_eval(self, result: EvalResult, is_best: bool) -> None:
        pass

    def on_train_end(self, curve: list[tuple[int, float, float]]) -> None:
        pass


class MetricSink(Protocol):
    def appendMetric(self, record: dict[str, object]) -> Path | None:
        ...


@dataclass(frozen=True)
class CheckpointContext:
    saveCheckpoint: Callable[[int, str, dict[str, object] | None], None]
    ckptPath: str
    latestPath: str
    snapshotPath: Callable[[int], str]
    pruneSnapshots: Callable[[], None]
    snapshotInterval: int
    bestValLoss: Callable[[], float | None]


# ---------------------------------------------------------------------------
# Concrete callbacks
# ---------------------------------------------------------------------------

class LoggingCallback(TrainingCallback):
    """Logs evaluation results and training-end summary."""

    def __init__(self, trainConfig: TrainConfig, logger: logging.Logger) -> None:
        self.trainConfig = trainConfig
        self.logger = logger

    def on_eval(self, result: EvalResult, is_best: bool) -> None:
        self.logger.info(
            "[step %s] train loss %.4f, val loss %.4f",
            result.step, result.train_loss, result.val_loss,
        )
        if result.frac_improvement is not None:
            self.logger.info(
                "[step %s] fractional improvement: %.4f (need > %.4f)",
                result.step, result.frac_improvement, self.trainConfig.earlyStopDelta,
            )
        if is_best:
            self.logger.info(
                "[step %s] New best val loss: %.4f",
                result.step, result.val_loss,
            )
        elif not result.improved:
            self.logger.info(
                "[step %s] No significant val improvement for %s evals.",
                result.step, result.no_improve_evals,
            )

    def on_train_end(self, curve: list[tuple[int, float, float]]) -> None:
        self.logger.info("Last few evals (step, train, val):")
        for step, tr, va in curve[-5:]:
            self.logger.info("  %6d: %.4f, %.4f", step, tr, va)


class MetricsCallback(TrainingCallback):
    """Appends evaluation results to metrics.jsonl via RunArtifacts."""

    def __init__(self, runArtifacts: MetricSink) -> None:
        self._runArtifacts = runArtifacts

    def on_eval(self, result: EvalResult, is_best: bool) -> None:
        self._runArtifacts.appendMetric(
            {
                "type": "evaluation",
                "step": result.step,
                "train_loss": result.train_loss,
                "validation_loss": result.val_loss,
                "fractional_improvement": result.frac_improvement,
                "improved": result.improved,
                "new_best": is_best,
                "no_improve_evals": result.no_improve_evals,
            }
        )


class CheckpointCallback(TrainingCallback):
    """Saves best.pt, latest.pt, and periodic snapshots after each evaluation."""

    def __init__(
        self,
        context: CheckpointContext,
        logger: logging.Logger,
    ) -> None:
        self._context = context
        self.logger = logger

    def on_eval(self, result: EvalResult, is_best: bool) -> None:
        context = self._context
        step = result.step

        if is_best:
            # best.pt gets a clean patience counter so resume always starts
            # fresh from the best model rather than inheriting stale patience.
            context.saveCheckpoint(
                step,
                context.ckptPath,
                {
                    "noImproveEvals": 0,
                    "referenceLoss": context.bestValLoss(),
                },
            )
            self.logger.info(
                "[step %s] Best checkpoint saved to %s.",
                step,
                context.ckptPath,
            )
        context.saveCheckpoint(step, context.latestPath, None)

        interval = context.snapshotInterval
        if interval > 0 and step > 0 and step % interval == 0:
            context.saveCheckpoint(step, context.snapshotPath(step), None)
            context.pruneSnapshots()


class TrainingCurveCallback(TrainingCallback):
    """Plots the training curve at the end of the run."""

    def __init__(self, modelConfig: ModelConfig, trainConfig: TrainConfig, logger: logging.Logger) -> None:
        self.modelConfig = modelConfig
        self.trainConfig = trainConfig
        self.logger = logger

    def on_train_end(self, curve: list[tuple[int, float, float]]) -> None:
        if not self.trainConfig.plotCurve:
            self.logger.info("Plotting disabled by config.")
            return
        if not curve:
            self.logger.info("No training curve data to plot.")
            return
        try:
            from .plot_utils import plot_training_curve
            filepath, config_dump_path = plot_training_curve(curve, self.modelConfig, self.trainConfig)
            self.logger.info("[plot] Saved plot to %s", filepath)
            self.logger.info("[plot] Saved config to %s", config_dump_path)
        except Exception as e:
            self.logger.warning("Could not plot training curve: %s", e)

