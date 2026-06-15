from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EarlyStopResult:
    improved: bool
    frac_improvement: Optional[float]
    should_stop: bool
    no_improve_evals: int


class EarlyStopping:
    def __init__(self, patience: int, delta: float) -> None:
        self.patience = patience
        self.delta = delta
        self.noImproveEvals = 0
        self.referenceLoss: Optional[float] = None

    def is_exhausted(self) -> bool:
        return self.noImproveEvals >= self.patience

    def reset(self) -> None:
        self.noImproveEvals = 0
        self.referenceLoss = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "noImproveEvals": self.noImproveEvals,
            "referenceLoss": self.referenceLoss,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.noImproveEvals = int(state.get("noImproveEvals", 0))
        reference = state.get("referenceLoss")
        self.referenceLoss = float(reference) if reference is not None else None

    def check(self, bestValLoss: Optional[float], currentValueLoss: float) -> EarlyStopResult:
        referenceLoss = self.referenceLoss
        if referenceLoss is None and bestValLoss is not None:
            referenceLoss = bestValLoss

        if referenceLoss is None or referenceLoss <= 0:
            fracImprovement = None
            improved = True
        else:
            fracImprovement = (
                referenceLoss - currentValueLoss
            ) / referenceLoss
            improved = fracImprovement > self.delta

        if improved:
            self.noImproveEvals = 0
            self.referenceLoss = currentValueLoss
        else:
            self.noImproveEvals += 1

        return EarlyStopResult(
            improved=improved,
            frac_improvement=fracImprovement,
            should_stop=self.noImproveEvals >= self.patience,
            no_improve_evals=self.noImproveEvals,
        )
