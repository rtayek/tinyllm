# Trainer.py

from __future__ import annotations

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging
import math

import torch

from Config import ModelConfig, TrainConfig
from Model import TinyGpt
from DataModule import ByteDataModule
from Checkpoints import CheckpointManager, CHECKPOINT_VERSION
from tensor_utils import tensor_to_int_list
from plot_utils import plot_training_curve


class LRScheduleStrategy:
    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, float]:
        raise NotImplementedError

    def load_state_dict(self, state: Dict[str, float]) -> None:
        raise NotImplementedError

    def align_after_resume(self, step: int) -> None:
        raise NotImplementedError


class WarmupCosineStrategy(LRScheduleStrategy):
    def __init__(self, optimizer: torch.optim.Optimizer, max_steps: int, warmup_frac: float) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_frac * max_steps))
        self.total_steps = max(max_steps, self.warmup_steps + 1)

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step + 1) / float(self.warmup_steps)

            progress = (current_step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
        )

    def step(self) -> None:
        self.scheduler.step()

    def state_dict(self) -> Dict[str, float]:
        return self.scheduler.state_dict()

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.scheduler.load_state_dict(state)

    def align_after_resume(self, step: int) -> None:
        if step > 0:
            self.scheduler.last_epoch = step - 1
            self.scheduler.step()


class EarlyStopping:
    def __init__(self, patience: int, delta: float) -> None:
        self.patience = patience
        self.delta = delta
        self.no_improve_evals = 0

    def reset(self) -> None:
        self.no_improve_evals = 0

    def check(
        self, best_val_loss: Optional[float], current_val_loss: float
    ) -> Tuple[bool, Optional[float], bool, int]:
        if best_val_loss is None or best_val_loss <= 0:
            frac_improvement = None
            improved = True
        else:
            frac_improvement = (best_val_loss - current_val_loss) / best_val_loss
            improved = frac_improvement > self.delta

        if improved:
            self.no_improve_evals = 0
        else:
            self.no_improve_evals += 1

        should_stop = self.no_improve_evals >= self.patience
        return improved, frac_improvement, should_stop, self.no_improve_evals


class Trainer:
    def __init__(
        self,
        modelCfg: ModelConfig,
        trainCfg: TrainConfig,
        model: TinyGpt,
        dataModule: ByteDataModule,
    ) -> None:
        self.modelCfg = modelCfg
        self.trainCfg = trainCfg
        self.model = model
        self.dataModule = dataModule
        self.logger = logging.getLogger(__name__)

        self.logger.info("MODEL CONFIG: %s", modelCfg)
        self.logger.info("TRAIN CONFIG: %s", trainCfg)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trainCfg.learningRate,
            weight_decay=trainCfg.weightDecay,
        )
        self.lrStrategy = WarmupCosineStrategy(
            self.optimizer,
            max_steps=trainCfg.maxSteps,
            warmup_frac=trainCfg.warmupFrac,
        )
        self.earlyStopping = EarlyStopping(trainCfg.earlyStopPatience, trainCfg.earlyStopDelta)
        self.checkpoints = CheckpointManager(modelCfg, trainCfg)

        self.globalStep: int = 0
        self.bestValLoss: Optional[float] = None
        self.trainingCurve: List[Tuple[int, float, float]] = []

        self.generator = torch.Generator()
        self.generator.manual_seed(1337)

    def _train_step(self) -> float:
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

    def _save_checkpoint(self, step: int) -> None:
        if self.bestValLoss is None:
            return
        self.checkpoints.save(
            self.model,
            self.optimizer,
            step,
            self.bestValLoss,
            self.lrStrategy.state_dict(),
        )
        self.logger.info("[step %s] Checkpoint saved (improved validation loss).", step)

    def _tensor_to_int_list(self, tensor: torch.Tensor) -> List[int]:
        return tensor_to_int_list(tensor)

    def _log_eval(self, step: int, evalResult: "Trainer.EvalResult") -> None:
        self.logger.info(
            "[step %s] train loss %.4f, val loss %.4f",
            step,
            evalResult.train_loss,
            evalResult.val_loss,
        )

        if evalResult.frac_improvement is not None:
            self.logger.info(
                "[step %s] fractional improvement: %.4f (need > %.4f)",
                step,
                evalResult.frac_improvement,
                self.trainCfg.earlyStopDelta,
            )

    @dataclass
    class EvalResult:
        step: int
        train_loss: float
        val_loss: float
        frac_improvement: Optional[float]
        improved: bool
        should_stop: bool
        no_improve_evals: int

    def evaluate(self, step: int) -> "Trainer.EvalResult":
        losses = self.estimateLoss()
        trainLoss = losses["train"]
        valLoss = losses["val"]

        improved, frac_improvement, should_stop, no_improve = self.earlyStopping.check(
            self.bestValLoss, valLoss
        )

        return Trainer.EvalResult(
            step=step,
            train_loss=trainLoss,
            val_loss=valLoss,
            frac_improvement=frac_improvement,
            improved=improved,
            should_stop=should_stop,
            no_improve_evals=no_improve,
        )

    def loadCheckpointIfExists(self) -> None:
        step, best, lr_state_restored, version, version_matches = self.checkpoints.load(
            self.model, self.optimizer, self.lrStrategy
        )
        self.globalStep = step
        self.bestValLoss = best
        if not lr_state_restored:
            self.lrStrategy.align_after_resume(step)
        if not version_matches:
            self.logger.warning(
                "Checkpoint version %s does not match expected %s; LR state not restored.",
                version,
                CHECKPOINT_VERSION,
            )
        else:
            self.logger.info("Loaded checkpoint version %s", version)
        self.earlyStopping.reset()

    def estimateLoss(self) -> Dict[str, float]:
        self.model.eval()
        losses: Dict[str, float] = {}

        with torch.no_grad():
            for split in ("train", "val"):
                lossList: List[float] = []
                for _ in range(self.trainCfg.evalIters):
                    batchX, batchY = self.dataModule.getBatch(split, self.generator)
                    _, loss = self.model(batchX, batchY)
                    if loss is None:
                        raise RuntimeError("Loss is None in estimateLoss")
                    lossList.append(float(loss.item()))
                losses[split] = sum(lossList) / float(len(lossList))

        self.model.train()
        return losses

    def train(self) -> None:
        self.logger.info("Using device: %s", self.trainCfg.device)
        self.logger.info("Starting training loop...")

        for step in range(self.globalStep, self.trainCfg.maxSteps):
            self.globalStep = step

            # ---- Evaluation ----
            if step % self.trainCfg.evalInterval == 0:
                self.logger.info("[step %s] Running evaluation...", step)

                evalResult = self.evaluate(step)
                self.trainingCurve.append((step, evalResult.train_loss, evalResult.val_loss))
                self._log_eval(step, evalResult)

                if evalResult.improved:
                    self.bestValLoss = evalResult.val_loss
                    self._save_checkpoint(step)
                else:
                    self.logger.info(
                        "[step %s] No val improvement for %s evals.",
                        step,
                        evalResult.no_improve_evals,
                    )

                    if evalResult.should_stop:
                        self.logger.info(
                            "[step %s] Early stopping triggered: no val improvement for %s evals.",
                            step,
                            evalResult.no_improve_evals,
                        )
                        break

            # ---- Training step ----
            _ = self._train_step()

        self.logger.info("Training loop finished.")
        self.logger.info("Best validation loss: %s", self.bestValLoss)
        self.logger.info(
            "Training done. Best val loss %.4f reached at some earlier step (see checkpoint metadata).",
            self.bestValLoss,
        )
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
            filepath, config_dump_path = plot_training_curve(
                self.trainingCurve,
                self.modelCfg,
                self.trainCfg,
            )
            self.logger.info("[plot] Saved plot to %s", filepath)
            self.logger.info("[plot] Saved config to %s", config_dump_path)
        except Exception as e:
            self.logger.info("Could not plot training curve: %s", e)

    def printSample(self, maxNewTokens: int = 200) -> None:
        start = torch.zeros((1, 1), dtype=torch.long, device=self.trainCfg.device)

        with torch.no_grad():
            generated: torch.Tensor = self.model.generate(start, maxNewTokens=maxNewTokens)

        firstSeq: torch.Tensor = generated[0]
        rawList = self._tensor_to_int_list(firstSeq.to(dtype=torch.long).view(-1))
        outBytes = bytes(rawList)
        decoded = outBytes.decode("utf-8", errors="ignore")

        self.logger.info("Sampled text:")
        self.logger.info(decoded)
