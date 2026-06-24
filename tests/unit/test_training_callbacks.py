"""Unit tests for TrainingCallback implementations."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

from llm.Config import ModelConfig, TrainConfig
from llm.Evaluator import EvalResult
from llm.TrainingCallback import (
    CheckpointCallback,
    LoggingCallback,
    MetricsCallback,
    TrainingCurveCallback,
)


def _result(
    step: int = 0,
    train_loss: float = 1.5,
    val_loss: float = 1.6,
    frac_improvement: float | None = 0.05,
    improved: bool = True,
    should_stop: bool = False,
    no_improve_evals: int = 0,
) -> EvalResult:
    return EvalResult(
        step=step,
        train_loss=train_loss,
        val_loss=val_loss,
        frac_improvement=frac_improvement,
        improved=improved,
        should_stop=should_stop,
        no_improve_evals=no_improve_evals,
    )


# ---------------------------------------------------------------------------
# LoggingCallback
# ---------------------------------------------------------------------------

class TestLoggingCallback:
    def setup_method(self) -> None:
        self.train_config = TrainConfig(device="cpu", earlyStopDelta=0.003)
        self.logger = MagicMock(spec=logging.Logger)
        self.cb = LoggingCallback(self.train_config, self.logger)

    def test_logs_losses_on_eval(self) -> None:
        self.cb.on_eval(_result(step=10, train_loss=1.5, val_loss=1.6), is_best=False)
        self.logger.info.assert_any_call(
            "[step %s] train loss %.4f, val loss %.4f", 10, 1.5, 1.6
        )

    def test_logs_fractional_improvement(self) -> None:
        self.cb.on_eval(_result(frac_improvement=0.05), is_best=False)
        calls = [str(c) for c in self.logger.info.call_args_list]
        assert any("fractional improvement" in c for c in calls)

    def test_logs_new_best_when_is_best(self) -> None:
        self.cb.on_eval(_result(step=5, val_loss=1.2), is_best=True)
        calls = [str(c) for c in self.logger.info.call_args_list]
        assert any("New best val loss" in c for c in calls)

    def test_logs_no_improvement_when_not_improved_and_not_best(self) -> None:
        self.cb.on_eval(_result(improved=False, no_improve_evals=1), is_best=False)
        calls = [str(c) for c in self.logger.info.call_args_list]
        assert any("No significant val improvement" in c for c in calls)

    def test_on_train_end_logs_curve(self) -> None:
        curve = [(0, 1.8, 1.9), (100, 1.5, 1.6)]
        self.cb.on_train_end(curve)
        calls = [str(c) for c in self.logger.info.call_args_list]
        assert any("Last few evals" in c for c in calls)

    def test_on_train_end_empty_curve(self) -> None:
        self.cb.on_train_end([])
        self.logger.info.assert_called()


# ---------------------------------------------------------------------------
# MetricsCallback
# ---------------------------------------------------------------------------

class TestMetricsCallback:
    def setup_method(self) -> None:
        self.run_artifacts = MagicMock()
        self.cb = MetricsCallback(self.run_artifacts)

    def test_appends_evaluation_metric(self) -> None:
        r = _result(step=10, train_loss=1.5, val_loss=1.6, frac_improvement=0.05,
                    improved=True, no_improve_evals=0)
        self.cb.on_eval(r, is_best=True)
        self.run_artifacts.appendMetric.assert_called_once()
        payload = self.run_artifacts.appendMetric.call_args[0][0]
        assert payload["type"] == "evaluation"
        assert payload["step"] == 10
        assert payload["new_best"] is True
        assert payload["train_loss"] == 1.5
        assert payload["validation_loss"] == 1.6

    def test_on_train_end_does_nothing(self) -> None:
        self.cb.on_train_end([(0, 1.0, 1.1)])
        self.run_artifacts.appendMetric.assert_not_called()


# ---------------------------------------------------------------------------
# CheckpointCallback
# ---------------------------------------------------------------------------

class TestCheckpointCallback:
    def _make_trainer(self, snapshot_interval: int = 0) -> MagicMock:
        trainer = MagicMock()
        trainer.trainConfig.snapshotInterval = snapshot_interval
        trainer.bestValLoss = 1.5
        trainer.checkpoints.ckptPath = "runs/exp/checkpoints/best.pt"
        trainer.checkpoints.latestPath = "runs/exp/checkpoints/latest.pt"
        trainer.checkpoints.snapshotPath.return_value = "runs/exp/checkpoints/step-000010.pt"
        return trainer

    def test_saves_best_and_latest_when_improved(self) -> None:
        trainer = self._make_trainer()
        cb = CheckpointCallback(trainer, MagicMock())
        cb.on_eval(_result(step=10), is_best=True)
        assert trainer._saveCheckpoint.call_count == 2
        paths = [c[0][1] for c in trainer._saveCheckpoint.call_args_list]
        assert "best.pt" in paths[0]
        assert "latest.pt" in paths[1]

    def test_saves_only_latest_when_not_best(self) -> None:
        trainer = self._make_trainer()
        cb = CheckpointCallback(trainer, MagicMock())
        cb.on_eval(_result(step=10), is_best=False)
        assert trainer._saveCheckpoint.call_count == 1
        assert "latest.pt" in trainer._saveCheckpoint.call_args[0][1]

    def test_saves_snapshot_at_interval(self) -> None:
        trainer = self._make_trainer(snapshot_interval=10)
        cb = CheckpointCallback(trainer, MagicMock())
        cb.on_eval(_result(step=10), is_best=False)
        assert trainer._saveCheckpoint.call_count == 2
        trainer.checkpoints.pruneSnapshots.assert_called_once()

    def test_no_snapshot_at_step_zero(self) -> None:
        trainer = self._make_trainer(snapshot_interval=1)
        cb = CheckpointCallback(trainer, MagicMock())
        cb.on_eval(_result(step=0), is_best=False)
        assert trainer._saveCheckpoint.call_count == 1

    def test_best_checkpoint_gets_clean_patience_counter(self) -> None:
        trainer = self._make_trainer()
        cb = CheckpointCallback(trainer, MagicMock())
        cb.on_eval(_result(step=5), is_best=True)
        early_stopping_state = trainer._saveCheckpoint.call_args_list[0][1]["earlyStoppingState"]
        assert early_stopping_state == {"noImproveEvals": 0, "referenceLoss": trainer.bestValLoss}

    def test_on_train_end_does_nothing(self) -> None:
        trainer = self._make_trainer()
        cb = CheckpointCallback(trainer, MagicMock())
        cb.on_train_end([])
        trainer._saveCheckpoint.assert_not_called()


# ---------------------------------------------------------------------------
# TrainingCurveCallback
# ---------------------------------------------------------------------------

class TestTrainingCurveCallback:
    def test_on_eval_does_nothing(self) -> None:
        cb = TrainingCurveCallback(
            ModelConfig(), TrainConfig(device="cpu"), MagicMock()
        )
        cb.on_eval(_result(), is_best=False)  # should not raise

    def test_on_train_end_skips_when_disabled(self) -> None:
        logger = MagicMock(spec=logging.Logger)
        cb = TrainingCurveCallback(
            ModelConfig(), TrainConfig(plotCurve=False, device="cpu"), logger
        )
        cb.on_train_end([(0, 1.0, 1.1)])
        logger.info.assert_called_with("Plotting disabled by config.")

    def test_on_train_end_skips_empty_curve(self) -> None:
        logger = MagicMock(spec=logging.Logger)
        cb = TrainingCurveCallback(
            ModelConfig(), TrainConfig(plotCurve=True, device="cpu"), logger
        )
        cb.on_train_end([])
        logger.info.assert_called_with("No training curve data to plot.")
