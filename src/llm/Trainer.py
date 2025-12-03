from __future__ import annotations

from typing import Any, Optional, List, Tuple, cast
import logging

import torch

from llm.Config import ModelConfig, TrainConfig
from llm.Model import TinyGpt
from llm.DataModule import SequenceDataModule
from llm.Checkpoint import CheckpointManager, CHECKPOINT_VERSION
from llm.LRScheduleStrategy import WarmupCosineStrategy
from llm.EarlyStopping import EarlyStopping
from llm.Evaluator import Evaluator # Import EvalResult and Evaluator directly


class Trainer:
    def __init__(
        self, modelConfig: ModelConfig, trainConfig: TrainConfig, model: TinyGpt, dataModule: SequenceDataModule, logger: Optional[logging.Logger] = None, evaluator: Optional[Evaluator] = None
    ) -> None:
        self.modelConfig = modelConfig
        self.trainConfig = trainConfig
        self.model = model
        self.dataModule = dataModule
        self.logger = logger or logging.getLogger(__name__)
        self.evaluator = evaluator

        self.logger.info("MODEL CONFIG: %s", self.modelConfig)
        self.logger.info("TRAIN CONFIG: %s", self.trainConfig)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=self.trainConfig.learningRate, weight_decay=self.trainConfig.weightDecay)
        assert self.trainConfig.batchSize > 0
        assert self.modelConfig.blockSize > 0
        assert self.trainConfig.learningRate > 0
        assert 0 <= self.trainConfig.warmupFrac <= 1
        self.lrStrategy: WarmupCosineStrategy = WarmupCosineStrategy(self.optimizer, max_steps=self.trainConfig.maxSteps, warmup_frac=self.trainConfig.warmupFrac)
        self.earlyStopping: EarlyStopping = EarlyStopping(self.trainConfig.earlyStopPatience, self.trainConfig.earlyStopDelta)
        self.checkpoints = CheckpointManager(self.modelConfig, self.trainConfig, logger=self.logger)

        self.globalStep: int = 0
        self.bestValLoss: Optional[float] = None
        self.trainingCurve: List[Tuple[int, float, float]] = []

        self.generator: torch.Generator = torch.Generator()
        self.generator.manual_seed(1337)


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

    def _saveCheckpoint(self, step: int) -> None:
        if self.bestValLoss is None:
            return
        self.checkpoints.saveCheckpoint(
            self.model,
            self.optimizer,
            self.lrStrategy.state_dict(),
            step,
            self.bestValLoss,
            generatorState=self.generator.get_state(),
        )
        self.logger.info("[step %s] Checkpoint saved (improved validation loss).", step)

    def _log_eval(self, step: int, evalResult: Any) -> None:
        self.logger.info("[step %s] train loss %.4f, val loss %.4f", step, evalResult.train_loss, evalResult.val_loss)

        if evalResult.frac_improvement is not None:
            self.logger.info("[step %s] fractional improvement: %.4f (need > %.4f)", step, evalResult.frac_improvement, self.trainConfig.earlyStopDelta)

    def loadCheckpointIfExists(self) -> None:
        (
            step,
            best,
            lrStateRestored,
            version,
            version_matches,
            config_drift,
            generator_state,
        ) = self.checkpoints.loadCheckpoint(
            self.model,
            self.optimizer,
            self.lrStrategy,
        )
        self.globalStep = step
        self.bestValLoss = best
        if generator_state is not None:
            self.generator.set_state(generator_state)
        if not lrStateRestored:
            self.lrStrategy.align_after_resume(step)
        if not version_matches:
            self.logger.warning(
                "Checkpoint version %s does not match expected %s; LR state not restored.",
                version,
                CHECKPOINT_VERSION,
            )
        else:
            self.logger.info("Loaded checkpoint version %s", version)
        if config_drift.get("model"):
            self.logger.warning("Model config drift from checkpoint: %s", config_drift["model"])
        if config_drift.get("train"):
            self.logger.warning("Train config drift from checkpoint: %s", config_drift["train"])
        self.earlyStopping.reset()

    def train(self) -> None:
        self.logger.info("Using device: %s", self.trainConfig.device)
        self.logger.info("Starting training loop...")

        for step in range(self.globalStep, self.trainConfig.maxSteps):
            self.globalStep = step

            if step % self.trainConfig.evalInterval == 0:
                self.logger.info("[step %s] Running evaluation...", step)

                if self.evaluator is None:
                    raise RuntimeError("Evaluator is not set.")
                evalResult: Any = self.evaluator.evaluate(step, self.bestValLoss)
                train_loss = float(cast(float, evalResult.train_loss))
                val_loss = float(cast(float, evalResult.val_loss))
                if not torch.isfinite(torch.tensor(train_loss)) or not torch.isfinite(
                    torch.tensor(val_loss)
                ):
                    raise RuntimeError("Non-finite evaluation loss encountered")
                self.trainingCurve.append((step, train_loss, val_loss))
                self._log_eval(step, evalResult)

                if bool(evalResult.improved):
                    self.bestValLoss = val_loss
                    self._saveCheckpoint(step)
                else:
                    self.logger.info("[step %s] No val improvement for %s evals.", step, evalResult.no_improve_evals)

                    if bool(evalResult.should_stop):
                        self.logger.info("[step %s] Early stopping triggered: no val improvement for %s evals.", step, evalResult.no_improve_evals)
                        break

            lossValue = self._trainStep()
            if not torch.isfinite(torch.tensor(lossValue)):
                raise RuntimeError("Non-finite training loss encountered")

        self.logger.info("Training loop finished.")
        if self.bestValLoss is not None:
            self.logger.info("Best validation loss: %s", self.bestValLoss)
            self.logger.info("Training done. Best val loss %.4f reached at some earlier step (see checkpoint metadata).", self.bestValLoss)
        else:
            self.logger.info("No validation loss recorded; training exited before evaluation.")
        self.logger.info("Last few evals (step, train, val):")
        for step, tr, va in self.trainingCurve[-5:]:
            self.logger.info("  %6d: %.4f, %.4f", step, tr, va)

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
        from .TextGenerator import TextGenerator

        generator = TextGenerator(self.model, self.trainConfig.device, self.logger)
        generator.log_sample(maxNewTokens=maxNewTokens, prompt=prompt)
