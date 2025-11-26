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
        warmupSteps = max(1, int(0.1 * cfg.maxSteps))
        totalSteps = max(cfg.maxSteps, warmupSteps + 1)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmupSteps:
                return float(current_step + 1) / float(warmupSteps)

            progress = (current_step - warmupSteps) / float(max(1, totalSteps - warmupSteps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
        )
        self.checkpoints = CheckpointManager(cfg)

        self.globalStep: int = 0
        self.bestValLoss: Optional[float] = None
        self.trainingCurve: List[Tuple[int, float, float]] = []

        self.generator = torch.Generator()
        self.generator.manual_seed(1337)

        self.noImproveEvals: int = 0

    @dataclass
    class EvalResult:
        step: int
        train_loss: float
        val_loss: float
        frac_improvement: Optional[float]
        improved: bool

    def evaluate(self, step: int) -> "Trainer.EvalResult":
        losses = self.estimateLoss()
        trainLoss = losses["train"]
        valLoss = losses["val"]

        if self.bestValLoss is None or self.bestValLoss <= 0:
            frac_improvement = None
            improved = True
        else:
            frac_improvement = (self.bestValLoss - valLoss) / self.bestValLoss
            improved = frac_improvement > self.cfg.earlyStopDelta

        return Trainer.EvalResult(
            step=step,
            train_loss=trainLoss,
            val_loss=valLoss,
            frac_improvement=frac_improvement,
            improved=improved,
        )

    def loadCheckpointIfExists(self) -> None:
        step, best, schedulerRestored = self.checkpoints.load(
            self.model, self.optimizer, self.scheduler
        )
        self.globalStep = step
        self.bestValLoss = best
        if not schedulerRestored and step > 0:
            self.scheduler.last_epoch = step - 1
            self.scheduler.step()

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

        # Early stopping counters
        self.noImproveEvals = 0

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
                    self.noImproveEvals = 0

                    self.checkpoints.save(
                        self.model,
                        self.optimizer,
                        step,
                        self.bestValLoss,
                        self.scheduler,
                    )

                    print(
                        f"[step {step}] Checkpoint saved (improved validation loss).",
                        flush=True,
                    )
                else:
                    self.noImproveEvals += 1
                    print(
                        f"[step {step}] No val improvement for "
                        f"{self.noImproveEvals} evals.",
                        flush=True,
                    )

                    if self.noImproveEvals >= self.cfg.earlyStopPatience:
                        print(
                            f"[step {step}] Early stopping triggered: "
                            f"no val improvement for "
                            f"{self.noImproveEvals} evals.",
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
            self.scheduler.step()

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
