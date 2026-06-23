from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
import math
import torch


class LRScheduleStrategy(ABC):
    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def state_dict(self) -> Dict[str, float]: ...

    @abstractmethod
    def load_state_dict(self, state: Dict[str, float]) -> None: ...

    @abstractmethod
    def align_after_resume(self, step: int) -> None: ...


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

        # Use positional arg to satisfy type checkers; LambdaLR supports callable scheduler.
        self.scheduler: Any = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def step(self) -> None:
        self.scheduler.step()

    def state_dict(self) -> Dict[str, float]:
        last = getattr(self.scheduler, "last_epoch", -1)
        return {"last_epoch": int(last)}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        last = state.get("last_epoch")
        if last is not None:
            self.align_after_resume(int(last) + 1)

    def align_after_resume(self, step: int) -> None:
        """Restore the scheduler to `step` after a checkpoint resume.

        Sets last_epoch directly on a freshly constructed scheduler (no prior
        step() calls), which is the safe use of this attribute and avoids the
        PyTorch warning about calling scheduler.step() before optimizer.step().
        """
        self.scheduler.last_epoch = step - 1
