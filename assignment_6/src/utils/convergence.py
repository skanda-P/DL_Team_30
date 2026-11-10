"""Successive-epoch training error convergence checker (|E_t - E_{t-1}| < 1e-4)."""
from __future__ import annotations

import math
from typing import Optional, Tuple


class ConvergenceChecker:
    """Stop training when |E_t - E_{t-1}| < tol (capped at max_epochs)."""

    def __init__(self, tol: float = 1e-4, min_epochs: int = 2, max_epochs: int = 1000) -> None:
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        if min_epochs < 2:
            raise ValueError(f"min_epochs must be >= 2, got {min_epochs}")
        if max_epochs < min_epochs:
            raise ValueError(f"max_epochs ({max_epochs}) must be >= min_epochs ({min_epochs})")

        self.tol = float(tol)
        self.min_epochs = int(min_epochs)
        self.max_epochs = int(max_epochs)
        self.prev_error: Optional[float] = None
        self.converged = False
        self.stop_reason: Optional[str] = None

    def step(self, epoch: int, train_error: float) -> Tuple[bool, Optional[str]]:
        """Check stopping condition at 1-based epoch `epoch`."""
        if not math.isfinite(train_error):
            self.converged = False
            self.stop_reason = "diverged"
            return True, self.stop_reason

        if self.prev_error is not None:
            diff = abs(train_error - self.prev_error)
            if epoch >= self.min_epochs and diff < self.tol:
                self.prev_error = train_error
                self.converged = True
                self.stop_reason = "converged"
                return True, self.stop_reason

        self.prev_error = train_error
        if epoch >= self.max_epochs:
            self.converged = False
            self.stop_reason = "max_epochs_cap"
            return True, self.stop_reason

        return False, None
