"""Training event callbacks.

The ``TrainingCallback`` protocol defines two lifecycle hooks that fire during
training.  Concrete callbacks implement one or both hooks and are registered on
``LMTrainer`` at construction time.  The trainer calls them in registration
order, which makes the full event flow visible in one place (``buildTrainer``
in ``Main.py``).

Hooks
-----
on_eval(result, is_best)
    Called after every evaluation pass.  ``is_best`` is True when ``result``
    produced a new absolute validation-loss minimum.

on_train_end(curve)
    Called once after the training loop exits (whether by exhausting steps,
    early stopping, or a non-finite loss error).  ``curve`` is the list of
    (step, train_loss, val_loss) tuples accumulated during the run.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

from .Config import ModelConfig, TrainConfig
from .Evaluator import EvalResult


class TrainingCallback(Protocol):
    def on_eval(self, result: EvalResult, is_best: bool) -> None: ...
    def on_train_end(self, curve: list[tuple[int, float, float]]) -> None: ...


# ---------------------------------------------------------------------------
# Concrete callbacks
# ---------------------------------------------------------------------------

class LoggingCallback:
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
                "[step %s] New best val loss: %.4f — checkpoint saved.",
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


class MetricsCallback:
    """Appends evaluation results to metrics.jsonl via RunArtifacts."""

    def __init__(self, runArtifacts: Any) -> None:
        # Typed as Any to avoid a circular import with RunArtifacts.
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

    def on_train_end(self, curve: list[tuple[int, float, float]]) -> None:
        pass


class CheckpointCallback:
    """Saves best.pt, latest.pt, and periodic snapshots after each evaluation."""

    def __init__(
        self,
        trainer: Any,  # LMTrainer — typed as Any to avoid circular import
        logger: logging.Logger,
    ) -> None:
        # Hold a reference to the trainer so we can reach its checkpoint
        # manager, optimizer, generator states, and early-stopping state.
        self._trainer = trainer
        self.logger = logger

    def on_eval(self, result: EvalResult, is_best: bool) -> None:
        trainer = self._trainer
        step = result.step

        if is_best:
            # best.pt gets a clean patience counter so resume always starts
            # fresh from the best model rather than inheriting stale patience.
            trainer._saveCheckpoint(
                step,
                trainer.checkpoints.ckptPath,
                earlyStoppingState={
                    "noImproveEvals": 0,
                    "referenceLoss": trainer.bestValLoss,
                },
            )
        trainer._saveCheckpoint(step, trainer.checkpoints.latestPath)

        interval = trainer.trainConfig.snapshotInterval
        if interval > 0 and step > 0 and step % interval == 0:
            trainer._saveCheckpoint(step, trainer.checkpoints.snapshotPath(step))
            trainer.checkpoints.pruneSnapshots()

    def on_train_end(self, curve: list[tuple[int, float, float]]) -> None:
        pass


class TrainingCurveCallback:
    """Plots the training curve at the end of the run."""

    def __init__(self, modelConfig: ModelConfig, trainConfig: TrainConfig, logger: logging.Logger) -> None:
        self.modelConfig = modelConfig
        self.trainConfig = trainConfig
        self.logger = logger

    def on_eval(self, result: EvalResult, is_best: bool) -> None:
        pass

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
            self.logger.info("Could not plot training curve: %s", e)
