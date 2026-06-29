from __future__ import annotations

import math
import os
from typing import Optional
import logging

import torch

from llm.Config import ModelConfig, TrainConfig
from llm.Model import TinyGPTLanguageModel
from llm.DataModule import SequenceDataModule
from llm.Checkpoint import Checkpoint, CheckpointManager, CheckpointLoadResult, CHECKPOINT_VERSION
from llm.LRScheduleStrategy import WarmupCosineStrategy
from llm.OptimizerFactory import (
    OptimizerFactory,
    SchedulerFactory,
    default_optimizer_factory,
    default_scheduler_factory,
)
from llm.Evaluator import Evaluator, TrainEvalResult
from llm.RunArtifacts import RunArtifacts
from llm.TrainingCallback import TrainingCallback


class LMTrainer:
    def __init__(
        self,
        modelConfig: ModelConfig,
        trainConfig: TrainConfig,
        model: TinyGPTLanguageModel,
        dataModule: SequenceDataModule,
        logger: Optional[logging.Logger] = None,
        evaluator: Optional[Evaluator] = None,
        callbacks: Optional[list[TrainingCallback]] = None,
        runArtifacts: Optional[RunArtifacts] = None,
        optimizerFactory: Optional[OptimizerFactory] = None,
        schedulerFactory: Optional[SchedulerFactory] = None,
    ) -> None:
        self.modelConfig = modelConfig
        self.trainConfig = trainConfig
        self.model = model
        self.dataModule = dataModule
        self.logger = logger or logging.getLogger(__name__)
        self.evaluator = evaluator
        self.callbacks: list[TrainingCallback] = callbacks or []

        self.logger.info("MODEL CONFIG: %s", self.modelConfig)
        self.logger.info("TRAIN CONFIG: %s", self.trainConfig)
        makeOptimizer: OptimizerFactory = optimizerFactory or default_optimizer_factory
        makeScheduler: SchedulerFactory = schedulerFactory or default_scheduler_factory
        self.optimizer: torch.optim.Optimizer = makeOptimizer(model, self.trainConfig)
        self.lrStrategy: WarmupCosineStrategy = makeScheduler(self.optimizer, self.trainConfig)
        self.checkpoints = CheckpointManager(self.modelConfig, self.trainConfig, logger=self.logger)
        self.runArtifacts = runArtifacts or RunArtifacts(self.modelConfig, self.trainConfig)

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

    def saveCheckpoint(
        self,
        step: int,
        path: str,
        earlyStoppingState: dict[str, object] | None = None,
    ) -> None:
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

    def _fire_on_eval(self, result: TrainEvalResult, is_best: bool) -> None:
        for cb in self.callbacks:
            cb.on_eval(result, is_best)

    def _fire_on_train_end(self) -> None:
        for cb in self.callbacks:
            cb.on_train_end(self.trainingCurve)

    def loadCheckpointIfExists(
        self,
        resetEarlyStopping: bool = False,
    ) -> None:
        resumePath = self.checkpoints.resumePath()
        checkpointExists = os.path.exists(resumePath)
        result: CheckpointLoadResult = self.checkpoints.restoreCheckpoint(
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

    def train(self) -> bool:
        self.logger.info("Using device: %s", self.trainConfig.device)
        if self.evaluator is not None and self.evaluator.early_stopping.is_exhausted():
            self.logger.info(
                "Training already stopped by early stopping at step %s. "
                "Use --reset-early-stopping to continue.",
                self.globalStep,
            )
            return False
        if self.globalStep >= self.trainConfig.maxSteps:
            self.logger.info(
                "Training already reached maxSteps=%s; no training work to do.",
                self.trainConfig.maxSteps,
            )
            return False
        self.logger.info("Starting training loop...")

        stoppedEarly = False
        for step in range(self.globalStep, self.trainConfig.maxSteps):
            self.globalStep = step

            if step % self.trainConfig.evalInterval == 0 and step != self._resumedFromStep:
                self.logger.info("[step %s] Running evaluation...", step)

                if self.evaluator is None:
                    raise RuntimeError("Evaluator is not set.")
                evalResult = self.evaluator.evaluate(step, self.bestValLoss)
                train_loss = float(evalResult.train_loss)
                val_loss = float(evalResult.val_loss)
                if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                    raise RuntimeError("Non-finite evaluation loss encountered")
                self.trainingCurve.append((step, train_loss, val_loss))

                isBest = self.bestValLoss is None or val_loss < self.bestValLoss
                if isBest:
                    self.bestValLoss = val_loss

                self._fire_on_eval(evalResult, isBest)

                if evalResult.should_stop:
                    self.logger.info(
                        "[step %s] Early stopping triggered: no val improvement for %s evals.",
                        step, evalResult.no_improve_evals,
                    )
                    stoppedEarly = True
                    break

            lossValue = self._trainStep()
            if not math.isfinite(lossValue):
                raise RuntimeError("Non-finite training loss encountered")
            self.globalStep = step + 1

        if not stoppedEarly and self.globalStep >= self.trainConfig.maxSteps:
            self.saveCheckpoint(self.globalStep, self.checkpoints.latestPath)

        self.logger.info("Training loop finished.")
        if self.bestValLoss is not None:
            self.logger.info(
                "Training done. Best val loss %.4f reached at some earlier step (see checkpoint metadata).",
                self.bestValLoss,
            )
        else:
            self.logger.info("No validation loss recorded; training exited before evaluation.")

        self._fire_on_train_end()
        return True

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

    def printSample(self, maxNewTokens: int = 200, prompt: str = "") -> None:
        from .TextGenerator import AutoregressiveGenerator

        generator = AutoregressiveGenerator(self.model, self.logger)
        generator.logSample(maxNewTokens=maxNewTokens, prompt=prompt)
