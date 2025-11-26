# Trainer.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingImports=false

from __future__ import annotations

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import math

import torch

from Config import ModelConfig
from Model import TinyGpt
from DataModule import ByteDataModule
from Checkpoints import CheckpointManager


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
        cfg: ModelConfig,
        model: TinyGpt,
        dataModule: ByteDataModule,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.dataModule = dataModule
        print("CONFIG DUMP:", cfg)
        self.optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.learningRate,weight_decay=cfg.weightDecay)
        self.lrStrategy = WarmupCosineStrategy(
            self.optimizer,
            max_steps=cfg.maxSteps,
            warmup_frac=cfg.warmupFrac,
        )
        self.earlyStopping = EarlyStopping(cfg.earlyStopPatience, cfg.earlyStopDelta)
        self.checkpoints = CheckpointManager(cfg)

        self.globalStep: int = 0
        self.bestValLoss: Optional[float] = None
        self.trainingCurve: List[Tuple[int, float, float]] = []

        self.generator = torch.Generator()
        self.generator.manual_seed(1337)

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
        step, best, schedulerRestored = self.checkpoints.load(
            self.model, self.optimizer, self.lrStrategy
        )
        self.globalStep = step
        self.bestValLoss = best
        if not schedulerRestored:
            self.lrStrategy.align_after_resume(step)
        self.earlyStopping.reset()

    def estimateLoss(self) -> Dict[str, float]:
        self.model.eval()
        losses: Dict[str, float] = {}

        with torch.no_grad():
            for split in ("train", "val"):
                lossList: List[float] = []
                for _ in range(self.cfg.evalIters):
                    batchX, batchY = self.dataModule.getBatch(split, self.generator)
                    _, loss = self.model(batchX, batchY)
                    if loss is None:
                        raise RuntimeError("Loss is None in estimateLoss")
                    lossList.append(float(loss.item()))
                losses[split] = sum(lossList) / float(len(lossList))

        self.model.train()
        return losses

    def train(self) -> None:
        print(f"Using device: {self.cfg.device}", flush=True)
        print("Starting training loop...", flush=True)

        for step in range(self.globalStep, self.cfg.maxSteps):
            self.globalStep = step

            # ---- Evaluation ----
            if step % self.cfg.evalInterval == 0:
                print(f"[step {step}] Running evaluation...", flush=True)

                evalResult = self.evaluate(step)
                self.trainingCurve.append((step, evalResult.train_loss, evalResult.val_loss))

                print(
                    f"[step {step}] train loss {evalResult.train_loss:.4f}, "
                    f"val loss {evalResult.val_loss:.4f}",
                    flush=True,
                )

                if evalResult.frac_improvement is not None:
                    print(
                        f"[step {step}] fractional improvement: {evalResult.frac_improvement:.4f} "
                        f"(need > {self.cfg.earlyStopDelta:.4f})",
                        flush=True,
                    )

                if evalResult.improved:
                    self.bestValLoss = evalResult.val_loss

                    self.checkpoints.save(
                        self.model,
                        self.optimizer,
                        step,
                        self.bestValLoss,
                        self.lrStrategy.state_dict(),
                    )

                    print(
                        f"[step {step}] Checkpoint saved (improved validation loss).",
                        flush=True,
                    )
                else:
                    print(
                        f"[step {step}] No val improvement for "
                        f"{evalResult.no_improve_evals} evals.",
                        flush=True,
                    )

                    if evalResult.should_stop:
                        print(
                            f"[step {step}] Early stopping triggered: "
                            f"no val improvement for "
                            f"{evalResult.no_improve_evals} evals.",
                            flush=True,
                        )
                        break

            # ---- Training step ----
            batchX, batchY = self.dataModule.getBatch("train", self.generator)
            _, loss = self.model(batchX, batchY)

            if loss is None:
                raise RuntimeError("Loss is None during training")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.lrStrategy.step()

        print("Training loop finished.", flush=True)
        print(f"Best validation loss: {self.bestValLoss}", flush=True)
        print(
            f"Training done. Best val loss {self.bestValLoss:.4f} "
            f"reached at some earlier step (see checkpoint metadata).",
            flush=True,
        )
        print("Last few evals (step, train, val):")
        for step, tr, va in self.trainingCurve[-5:]:
            print(f"  {step:6d}: {tr:.4f}, {va:.4f}")

        

    def plotTrainingCurve(self) -> None:
        if not self.trainingCurve:
            print("No trainingCurve data to plot.", flush=True)
            return

        try:
            import matplotlib.pyplot as plt  # type: ignore[import]
            import os
            from datetime import datetime

            # Create output directory
            out_dir = "plots"
            os.makedirs(out_dir, exist_ok=True)

            # Build a short name from config
            cfg = self.cfg
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Filename encodes key hyperparameters
            filename = (
                f"plot_lr{cfg.learningRate}_wd{cfg.weightDecay}"
                f"_do{cfg.dropout}_bs{cfg.batchSize}_"
                f"{timestamp}.png"
            )
            filepath = os.path.join(out_dir, filename)

            # Dump config to text file
            config_dump_path = os.path.join(
                out_dir, f"config_{timestamp}.txt"
            )
            with open(config_dump_path, "w", encoding="utf-8") as f:
                f.write("TRAINING CONFIGURATION:\n")
                for field, value in vars(cfg).items():
                    f.write(f"{field} = {value}\n")

            # Extract curve data
            steps = [x[0] for x in self.trainingCurve]
            trainLosses = [x[1] for x in self.trainingCurve]
            valLosses = [x[2] for x in self.trainingCurve]

            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(steps, trainLosses, label="train loss")
            plt.plot(steps, valLosses, label="val loss")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Training Curve")
            plt.legend()
            plt.grid(True)

            # Save plot
            plt.savefig(filepath, dpi=150)
            print(f"[plot] Saved plot to {filepath}", flush=True)
            print(f"[plot] Saved config to {config_dump_path}", flush=True)

            # Optionally show it live
            plt.show()

        except Exception as e:
            print(f"Could not plot training curve: {e}", flush=True)

    def printSample(self, maxNewTokens: int = 200) -> None:
        start = torch.zeros((1, 1), dtype=torch.long, device=self.cfg.device)

        with torch.no_grad():
            generated = self.model.generate(start, maxNewTokens=maxNewTokens)

        firstSeq = generated[0]
        rawList = firstSeq.tolist()
        outBytes = bytes(int(v) for v in rawList)
        decoded = outBytes.decode("utf-8", errors="ignore")

        print("\nSampled text:")
        print(decoded)
