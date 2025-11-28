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
