from __future__ import annotations

from typing import Optional, Tuple


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
