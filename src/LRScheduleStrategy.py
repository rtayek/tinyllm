from __future__ import annotations

from typing import Dict
import math
import torch


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
        self.warmupSteps = max(1, int(warmup_frac * max_steps))
        self.totalSteps = max(max_steps, self.warmupSteps + 1)

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmupSteps:
                return float(current_step + 1) / float(self.warmupSteps)

            progress = (current_step - self.warmupSteps) / float(
                max(1, self.totalSteps - self.warmupSteps)
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
            # Position the scheduler so the next step() call will produce LR for this step
            self.scheduler.last_epoch = step - 1
