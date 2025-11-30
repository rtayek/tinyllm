from __future__ import annotations

from typing import Optional, Tuple


class EarlyStopping:
    def __init__(self, patience: int, delta: float) -> None:
        self.patience = patience
        self.delta = delta
        self.noImproveEvals = 0

    def reset(self) -> None:
        self.noImproveEvals = 0

    def check(self, bestValLoss: Optional[float], currentValueLoss: float) -> Tuple[bool, Optional[float], bool, int]:
        if bestValLoss is None or bestValLoss <= 0:
            fracImprovement = None
            improved = True
        else:
            fracImprovement = (bestValLoss - currentValueLoss) / bestValLoss
            improved = fracImprovement > self.delta

        if improved:
            self.noImproveEvals = 0
        else:
            self.noImproveEvals += 1

        shouldStop = self.noImproveEvals >= self.patience
        return improved, fracImprovement, shouldStop, self.noImproveEvals
