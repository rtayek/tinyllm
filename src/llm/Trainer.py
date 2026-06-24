from __future__ import annotations

import math
import os
from typing import Any, Optional, cast
import logging

import torch

from llm.Config import ModelConfig, TrainConfig
from llm.Model import TinyGPTLanguageModel
from llm.DataModule import SequenceDataModule
from llm.Checkpoint import Checkpoint, CheckpointManager, CheckpointLoadResult, CHECKPOINT_VERSION
from llm.LRScheduleStrategy import WarmupCosineStrategy
from llm.Evaluator import Evaluator
from llm.RunArtifacts import RunArtifacts


class LMTrainer:
    def __init__(
        self,
        modelConfig: ModelConfig,
        trainConfig: TrainConfig,
        model: TinyGPTLanguageModel,
        dataModule: SequenceDataModule,
        logger: Optional[logging.Logger] = None,
        evaluator: Optional[Evaluator] = None,
    ) -> None:
        self.modelConfig = modelConfig
        self.trainConfig = trainConfig
        self.model = model
        self.dataModule = dataModule
        self.logger = logger or logging.getLogger(__name__)
        self.evaluator = evaluator

        self.logger.info("MODEL CONFIG: %s", self.modelConfig)
        self.logger.info("TRAIN CONFIG: %s", self.trainConfig)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.trainConfig.learningRate,
            weight_decay=self.trainConfig.weightDecay,
        )
        self.lrStrategy: WarmupCosineStrategy = WarmupCosineStrategy(
            self.optimizer,
            max_steps=self.trainConfig.maxSteps,
            warmup_frac=self.trainConfig.warmupFrac,
        )
        self.checkpoints = CheckpointManager(self.modelConfig, self.trainConfig, logger=self.logger)
        self.runArtifacts = RunArtifacts(self.modelConfig, self.trainConfig)
        metadataPath = self.runArtifacts.writeRunMetadata()
        if metadataPath is not None:
            self.logger.info("Run metadata written to %s", metadataPath)

        self.globalStep: int = 0
        self.bestValLoss: Optional[float] = None
        self.trainingCurve: list[tuple[int, float, float]] = []
        self._resumedFromStep: int = -1  # set by loadCheckpointIfExists to skip redundant eval on resume

        self.generator: torch.Generator = torch.Generator()
        self.generator.manual_seed(self.trainConfig.seed)


    def _trainStep(self) -> float:
        batchX, batchY = self.dataModule.getBatch("train", self.generator)
        _, loss, _ = self.model(batchX, batchY)

        if loss is None:
            raise RuntimeError("Loss is None during training")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.lrStrategy.step()

        return float(loss.item())

    def _saveCheckpoint(self, step: int, path: str, earlyStoppingState: Optional[dict[str, Any]] = None) -> None:
        if earlyStoppingState is None and self.evaluator is not None:
            earlyStoppingState = self.evaluator.early_stopping.state_dict()
        self.checkpoints.saveCheckpoint(
            self.model,
            self.optimizer,
            self.lrStrategy.state_dict(),
            step,
            self.bestValLoss,
            generatorState=self.generator.get_state(),
            evaluatorGeneratorState=(
                self.evaluator.generator.get_state()
                if self.evaluator is not None
                else None
            ),
            earlyStoppingState=earlyStoppingState,
            path=path,
        )
        self.logger.info("[step %s] Checkpoint saved to %s.", step, path)

    def _saveEvaluationCheckpoints(self, step: int, improved: bool) -> None:
        if improved:
            # best.pt gets a clean patience counter so resume always starts fresh from this model
            self._saveCheckpoint(
                step,
                self.checkpoints.ckptPath,
                earlyStoppingState={"noImproveEvals": 0, "referenceLoss": self.bestValLoss},
            )
        self._saveCheckpoint(step, self.checkpoints.latestPath)

        interval = self.trainConfig.snapshotInterval
        if interval > 0 and step > 0 and step % interval == 0:
            self._saveCheckpoint(step, self.checkpoints.snapshotPath(step))
            self.checkpoints.pruneSnapshots()

    def _log_eval(self, step: int, evalResult: Any) -> None:
        self.logger.info("[step %s] train loss %.4f, val loss %.4f", step, evalResult.train_loss, evalResult.val_loss)

        if evalResult.frac_improvement is not None:
            self.logger.info("[step %s] fractional improvement: %.4f (need > %.4f)", step, evalResult.frac_improvement, self.trainConfig.earlyStopDelta)

    def loadCheckpointIfExists(
        self,
        resetEarlyStopping: bool = False,
    ) -> None:
        resumePath = self.checkpoints.resumePath()
        checkpointExists = os.path.exists(resumePath)
        result: CheckpointLoadResult = self.checkpoints.loadCheckpoint(
            self.model,
            self.optimizer,
            self.lrStrategy,
        )
        self.globalStep = result.step
        self.bestValLoss = result.bestValLoss
        # Only skip the first eval if we actually resumed — a fresh run at step 0
        # should still evaluate before its first training step.
        if checkpointExists:
            self._resumedFromStep = result.step
        if result.generatorState is not None:
            self.generator.set_state(result.generatorState)
        if self.evaluator is not None:
            if result.evaluatorGeneratorState is not None:
                self.evaluator.generator.set_state(result.evaluatorGeneratorState)
            if result.earlyStoppingState is not None:
                self.evaluator.early_stopping.load_state_dict(result.earlyStoppingState)
            if resetEarlyStopping:
                self.evaluator.early_stopping.reset()
                self.logger.info("Restored early-stopping progress was reset.")
        if not result.lrStateRestored:
            self.lrStrategy.align_after_resume(result.step)
        if not checkpointExists:
            self.logger.info(
                "No checkpoint found at %s; starting a new run.",
                resumePath,
            )
        elif not result.versionMatches:
            self.logger.warning(
                "Checkpoint version %s does not match expected %s; LR state not restored.",
                result.version,
                CHECKPOINT_VERSION,
            )
        else:
            self.logger.info(
                "Loaded checkpoint version %s from %s; resuming at step %s",
                result.version,
                resumePath,
                result.step,
            )
        if result.configDrift.get("model"):
            self.logger.warning("Model config drift from checkpoint: %s", result.configDrift["model"])
        if result.configDrift.get("train"):
            self.logger.warning("Train config drift from checkpoint: %s", result.configDrift["train"])

    def train(self) -> None:
        self.logger.info("Using device: %s", self.trainConfig.device)
        if self.evaluator is not None and self.evaluator.early_stopping.is_exhausted():
            self.logger.info(
                "Training already stopped by early stopping at step %s. "
                "Use --reset-early-stopping to continue.",
                self.globalStep,
            )
            return
        self.logger.info("Starting training loop...")

        for step in range(self.globalStep, self.trainConfig.maxSteps):
            self.globalStep = step

            if step % self.trainConfig.evalInterval == 0 and step != self._resumedFromStep:
                self.logger.info("[step %s] Running evaluation...", step)

                if self.evaluator is None:
                    raise RuntimeError("Evaluator is not set.")
                evalResult: Any = self.evaluator.evaluate(step, self.bestValLoss)
                train_loss = float(cast(float, evalResult.train_loss))
                val_loss = float(cast(float, evalResult.val_loss))
                if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                    raise RuntimeError("Non-finite evaluation loss encountered")
                self.trainingCurve.append((step, train_loss, val_loss))
                self._log_eval(step, evalResult)
                isBest = self.bestValLoss is None or val_loss < self.bestValLoss
                self.runArtifacts.appendMetric(
                    {
                        "type": "evaluation",
                        "step": step,
                        "train_loss": train_loss,
                        "validation_loss": val_loss,
                        "fractional_improvement": evalResult.frac_improvement,
                        "improved": evalResult.improved,
                        "new_best": isBest,
                        "no_improve_evals": evalResult.no_improve_evals,
                    }
                )

                if isBest:
                    self.bestValLoss = val_loss
                    self.logger.info("[step %s] New best val loss: %.4f — checkpoint saved.", step, val_loss)
                if not evalResult.improved and not isBest:
                    self.logger.info(
                        "[step %s] No significant val improvement for %s evals.",
                        step,
                        evalResult.no_improve_evals,
                    )

                self._saveEvaluationCheckpoints(step, isBest)

                if evalResult.should_stop:
                    self.logger.info("[step %s] Early stopping triggered: no val improvement for %s evals.", step, evalResult.no_improve_evals)
                    break

            lossValue = self._trainStep()
            if not math.isfinite(lossValue):
                raise RuntimeError("Non-finite training loss encountered")

        self.logger.info("Training loop finished.")
        if self.bestValLoss is not None:
            self.logger.info("Training done. Best val loss %.4f reached at some earlier step (see checkpoint metadata).", self.bestValLoss)
        else:
            self.logger.info("No validation loss recorded; training exited before evaluation.")
        self.logger.info("Last few evals (step, train, val):")
        for step, tr, va in self.trainingCurve[-5:]:
            self.logger.info("  %6d: %.4f, %.4f", step, tr, va)

    def evaluateBestCheckpointOnTest(self) -> float | None:
        if self.evaluator is None or self.dataModule.testSequence is None:
            self.logger.info("No test split configured; skipping final test evaluation.")
            return None
        if not os.path.exists(self.checkpoints.ckptPath):
            self.logger.warning(
                "Best checkpoint not found at %s; skipping final test evaluation.",
                self.checkpoints.ckptPath,
            )
            return None

        checkpoint = Checkpoint.load(
            self.checkpoints.ckptPath,
            self.trainConfig.device,
        )
        self.model.load_state_dict(checkpoint.modelState)
        testGenerator = torch.Generator()
        testGenerator.manual_seed(self.trainConfig.seed + 2)
        testLoss = self.evaluator.estimate_split("test", testGenerator)
        self.logger.info(
            "Best checkpoint test loss at step %s: %.4f",
            checkpoint.step,
            testLoss,
        )
        self.runArtifacts.appendMetric(
            {
                "type": "test",
                "step": checkpoint.step,
                "test_loss": testLoss,
                "seed": self.trainConfig.seed + 2,
                "iterations": self.trainConfig.evalIters,
            }
        )
        return testLoss

    def plotTrainingCurve(self) -> None:
        if not self.trainConfig.plotCurve:
            self.logger.info("Plotting disabled by config.")
            return
        if not self.trainingCurve:
            self.logger.info("No trainingCurve data to plot.")
            return

        try:
            from .plot_utils import plot_training_curve

            filepath, config_dump_path = plot_training_curve(self.trainingCurve, self.modelConfig, self.trainConfig)
            self.logger.info("[plot] Saved plot to %s", filepath)
            self.logger.info("[plot] Saved config to %s", config_dump_path)
        except Exception as e:
            self.logger.info("Could not plot training curve: %s", e)

    def printSample(self, maxNewTokens: int = 200, prompt: str = "") -> None:
        from .TextGenerator import AutoregressiveGenerator

        generator = AutoregressiveGenerator(self.model, self.trainConfig.device, self.logger)
        generator.logSample(maxNewTokens=maxNewTokens, prompt=prompt)
