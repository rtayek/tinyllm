# Trainer.py

from __future__ import annotations

from typing import Optional, List, Tuple
import logging

import torch

from Config import ModelConfig, TrainConfig
from Model import TinyGpt
from DataModule import ByteDataModule
from Checkpoint import CheckpointManager, CHECKPOINT_VERSION
from LRScheduleStrategy import WarmupCosineStrategy
from EarlyStopping import EarlyStopping
from evaluator import Evaluator, EvalResult


class Trainer:
    def __init__(
        self, modelCfg: ModelConfig, trainCfg: TrainConfig, model: TinyGpt, dataModule: ByteDataModule, logger: Optional[logging.Logger] = None
    ) -> None:
        self.modelCfg = modelCfg
        self.trainCfg = trainCfg
        self.model = model
        self.dataModule = dataModule
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info("MODEL CONFIG: %s", modelCfg)
        self.logger.info("TRAIN CONFIG: %s", trainCfg)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=trainCfg.learningRate, weight_decay=trainCfg.weightDecay)
        assert trainCfg.batchSize > 0
        assert self.modelCfg.blockSize > 0
        assert trainCfg.learningRate > 0
        assert 0 <= trainCfg.warmupFrac <= 1
        self.lrStrategy = WarmupCosineStrategy(self.optimizer, max_steps=trainCfg.maxSteps, warmup_frac=trainCfg.warmupFrac)
        self.earlyStopping = EarlyStopping(trainCfg.earlyStopPatience, trainCfg.earlyStopDelta)
        self.checkpoints = CheckpointManager(modelCfg, trainCfg, logger=self.logger)

        self.globalStep: int = 0
        self.bestValLoss: Optional[float] = None
        self.trainingCurve: List[Tuple[int, float, float]] = []

        self.generator = torch.Generator()
        self.generator.manual_seed(1337)
        self.evaluator = Evaluator(self.model, self.dataModule, self.trainCfg, self.earlyStopping, self.generator, logger=self.logger)

    def _trainStep(self) -> float:
        batchX, batchY = self.dataModule.getBatch("train", self.generator)
        _, loss = self.model(batchX, batchY)

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

    def _log_eval(self, step: int, evalResult: EvalResult) -> None:
        self.logger.info("[step %s] train loss %.4f, val loss %.4f", step, evalResult.train_loss, evalResult.val_loss)

        if evalResult.frac_improvement is not None:
            self.logger.info("[step %s] fractional improvement: %.4f (need > %.4f)", step, evalResult.frac_improvement, self.trainCfg.earlyStopDelta)

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
        self.logger.info("Using device: %s", self.trainCfg.device)
        self.logger.info("Starting training loop...")

        for step in range(self.globalStep, self.trainCfg.maxSteps):
            self.globalStep = step

            # ---- Evaluation ----
            if step % self.trainCfg.evalInterval == 0:
                self.logger.info("[step %s] Running evaluation...", step)

                evalResult = self.evaluator.evaluate(step, self.bestValLoss)
                if not torch.isfinite(torch.tensor(evalResult.train_loss)) or not torch.isfinite(
                    torch.tensor(evalResult.val_loss)
                ):
                    raise RuntimeError("Non-finite evaluation loss encountered")
                self.trainingCurve.append((step, evalResult.train_loss, evalResult.val_loss))
                self._log_eval(step, evalResult)

                if evalResult.improved:
                    self.bestValLoss = evalResult.val_loss
                    self._saveCheckpoint(step)
                else:
                    self.logger.info("[step %s] No val improvement for %s evals.", step, evalResult.no_improve_evals)

                    if evalResult.should_stop:
                        self.logger.info("[step %s] Early stopping triggered: no val improvement for %s evals.", step, evalResult.no_improve_evals)
                        break

            # ---- Training step ----
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
        if not self.trainCfg.plotCurve:
            self.logger.info("Plotting disabled by config.")
            return
        if not self.trainingCurve:
            self.logger.info("No trainingCurve data to plot.")
            return

        try:
            from plot_utils import plot_training_curve

            filepath, config_dump_path = plot_training_curve(self.trainingCurve, self.modelCfg, self.trainCfg)
            self.logger.info("[plot] Saved plot to %s", filepath)
            self.logger.info("[plot] Saved config to %s", config_dump_path)
        except Exception as e:
            self.logger.info("Could not plot training curve: %s", e)

    def printSample(self, maxNewTokens: int = 200, prompt: str = "") -> None:
        from TextGenerator import TextGenerator

        generator = TextGenerator(self.model, self.trainCfg.device, self.logger)
        generator.log_sample(maxNewTokens=maxNewTokens, prompt=prompt)
