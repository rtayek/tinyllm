# Trainer.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingImports=false

from __future__ import annotations

from typing import Optional, Dict, List, Tuple

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
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learningRate)
        self.checkpoints = CheckpointManager(cfg)

        self.globalStep: int = 0
        self.bestValLoss: Optional[float] = None
        self.trainingCurve: List[Tuple[int, float, float]] = []

        self.generator = torch.Generator()
        self.generator.manual_seed(1337)

    def loadCheckpointIfExists(self) -> None:
        step, best = self.checkpoints.load(self.model, self.optimizer)
        self.globalStep = step
        self.bestValLoss = best

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

            if step % self.cfg.evalInterval == 0:
                print(f"[step {step}] Running evaluation...", flush=True)
                losses = self.estimateLoss()
                trainLoss = losses["train"]
                valLoss = losses["val"]

                print(
                    f"[step {step}] train loss {trainLoss:.4f}, "
                    f"val loss {valLoss:.4f}",
                    flush=True,
                )

                self.trainingCurve.append((step, trainLoss, valLoss))

                improved = self.bestValLoss is None or valLoss < self.bestValLoss
                if improved:
                    self.bestValLoss = valLoss
                    self.checkpoints.save(self.model, self.optimizer, step, self.bestValLoss)
                    print(
                        f"[step {step}] Checkpoint saved (improved validation loss).",
                        flush=True,
                    )

            batchX, batchY = self.dataModule.getBatch("train", self.generator)
            _, loss = self.model(batchX, batchY)
            if loss is None:
                raise RuntimeError("Loss is None during training")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if step % 10 == 0:
                print(
                    f"[step {step}] train batch loss {loss.item():.4f}",
                    flush=True,
                )

        self.checkpoints.save(self.model, self.optimizer, self.cfg.maxSteps, self.bestValLoss)
        print("Final checkpoint saved.", flush=True)

    def plotTrainingCurve(self) -> None:
        if not self.trainingCurve:
            print("No trainingCurve data to plot.", flush=True)
            return

        try:
            import matplotlib.pyplot as plt  # type: ignore[import]

            steps = [x[0] for x in self.trainingCurve]
            trainLosses = [x[1] for x in self.trainingCurve]
            valLosses = [x[2] for x in self.trainingCurve]

            plt.figure(figsize=(10, 5))
            plt.plot(steps, trainLosses, label="train loss")
            plt.plot(steps, valLosses, label="val loss")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Training Curve")
            plt.legend()
            plt.grid(True)
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
