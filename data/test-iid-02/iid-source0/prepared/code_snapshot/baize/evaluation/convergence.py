"""Convergence testing for training curves.

The original D-019 check asked whether the last N validation scores fell within a
fixed spread. That works for clean training, where the curve flattens smoothly,
and fails for noisy-label training, where the curve oscillates persistently
around a stable level and NEVER satisfies a spread bound however long it runs.
Every 20% and 40% noise run reported plateaued=False at spreads of 0.006-0.013
while being, in substance, done.

The right question is not "is the curve flat?" but "is it still going anywhere?"
So this fits a least-squares line to the tail of the curve and asks whether the
slope is distinguishable from zero given the curve's own scatter. Oscillation
inflates the scatter, which widens the slope's standard error, which correctly
makes the test MORE tolerant of a noisy-but-stationary curve rather than less.

Three outcomes, because they demand different responses:

    CONVERGED   slope indistinguishable from zero -> the run is done
    IMPROVING   slope significantly positive      -> extend it
    DEGRADING   slope significantly negative      -> the model is fitting noise;
                do NOT extend, and report the early-stopped round as a result

DEGRADING is not a failure mode to be fixed by more compute. At 40% label noise
the validation curve peaks around round 15 and declines thereafter, so a fixed
round budget applied across a noise sweep would systematically under-report high
noise settings and inflate any hypothesis claiming that noise hurts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConvergenceReport:
    status: str  # CONVERGED | IMPROVING | DEGRADING | INSUFFICIENT_DATA
    slope_per_round: float
    slope_stderr: float
    t_statistic: float
    window: int
    projected_gain: float  # extrapolated change over another `window` rounds
    best_index: int
    n_points: int
    best_at_cap: bool

    @property
    def converged(self) -> bool:
        return self.status == "CONVERGED"

    def render(self) -> str:
        return (
            f"convergence: {self.status}  "
            f"slope={self.slope_per_round:+.2e}/round (t={self.t_statistic:+.2f}), "
            f"projected gain over next {self.window} rounds "
            f"{self.projected_gain:+.4f}, best_at_cap={self.best_at_cap}"
        )


def check_convergence(
    values,
    window: int = 15,
    t_threshold: float = 2.0,
    negligible_gain: float = 0.002,
) -> ConvergenceReport:
    """Assess whether a training curve has stopped moving.

    `negligible_gain` guards against the large-sample trap: with a long, tight
    window a truly tiny slope can still be statistically significant. A slope is
    only called IMPROVING if it is BOTH statistically distinguishable from zero
    AND projects a gain worth the compute. Statistical significance alone is not
    a reason to keep training.
    """
    v = np.asarray(values, dtype=np.float64)
    if len(v) < max(4, window // 2):
        return ConvergenceReport("INSUFFICIENT_DATA", 0.0, 0.0, 0.0, window,
                                 0.0, int(np.argmax(v)) if len(v) else -1, len(v), False)

    tail = v[-window:]
    n = len(tail)
    x = np.arange(n, dtype=np.float64)

    x_c = x - x.mean()
    denom = float((x_c**2).sum())
    slope = float((x_c * (tail - tail.mean())).sum() / denom)
    intercept = float(tail.mean() - slope * x.mean())
    resid = tail - (intercept + slope * x)
    s2 = float((resid**2).sum() / max(n - 2, 1))
    stderr = float(np.sqrt(s2 / denom)) if denom > 0 else 0.0
    if stderr > 0:
        t = slope / stderr
    elif abs(slope) < 1e-12:
        t = 0.0
    else:
        t = float("inf") if slope > 0 else float("-inf")
    projected = slope * window

    if abs(t) < t_threshold or abs(projected) < negligible_gain:
        status = "CONVERGED"
    elif slope > 0:
        status = "IMPROVING"
    else:
        status = "DEGRADING"

    best_idx = int(np.argmax(v))
    return ConvergenceReport(
        status=status,
        slope_per_round=slope,
        slope_stderr=stderr,
        t_statistic=t,
        window=window,
        projected_gain=projected,
        best_index=best_idx,
        n_points=len(v),
        best_at_cap=bool(best_idx >= len(v) - 1),
    )
