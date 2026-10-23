"""Controlled label-noise injection with a validation gate.

Three processes, per THREAT_MODEL.md §3:
  symmetric          - flip with probability p regardless of class
  class_conditional  - per-class flip probability
  client_heterogeneous - per-client rate (applied by the caller, one client at a time)

Two rules enforced here:

1. Noise is applied to TRAINING labels only. Nothing in this module can touch a
   test set, because it operates on label arrays handed to it and the runner
   never hands it the test labels. The gate in `apply_noise` additionally
   refuses arrays tagged as protected.
2. The REALISED rate is measured and reported, never the target. If the two
   diverge beyond tolerance the run is BLOCKED (master spec §27). With small
   client shards the realised rate can drift from the target by a lot, and
   silently reporting the target would be a fabricated number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class NoiseValidationError(RuntimeError):
    """Raised when realised noise deviates from target beyond tolerance."""


def effective_tolerance(
    target: float, n: int, absolute_floor: float = 0.005, sigmas: float = 4.0
) -> float:
    """Sample-size-aware tolerance for the realised-vs-target noise check.

    A FIXED absolute tolerance is the wrong statistic. Flipping labels is a
    binomial process, so the realised rate has standard error sqrt(p(1-p)/n).
    At n=800 and p=0.4 that is ~0.017, meaning a fixed 0.02 tolerance rejects
    perfectly ordinary sampling variation on any small client shard - and
    tightening it further would make small-shard experiments impossible to run.

    The gate must catch a WRONG IMPLEMENTATION (a systematically off rate), not
    ordinary sampling noise. So the tolerance is `sigmas` standard errors, with
    an absolute floor to stay meaningful at very large n where the SE shrinks
    towards zero.

    sigmas=4 gives a false-block probability well under 0.1% per check if the
    injector is correct, while a genuine implementation error - flipping at the
    wrong rate, or double-flipping - shifts the mean by far more than 4 SE.
    """
    if n <= 0:
        return absolute_floor
    p = min(max(target, 0.0), 1.0)
    se = (p * (1.0 - p) / n) ** 0.5
    return max(absolute_floor, sigmas * se)


@dataclass
class NoiseReport:
    process: str
    target_rate: float
    realised_rate: float
    n_samples: int
    n_flipped: int
    seed: int
    tolerance: float
    per_class_realised: dict[int, float] = field(default_factory=dict)
    per_client: list[dict] = field(default_factory=list)
    status: str = "UNKNOWN"

    @property
    def deviation_sigmas(self) -> float:
        """How many binomial standard errors the realised rate sits from target."""
        p, n = self.target_rate, self.n_samples
        if n <= 0 or p <= 0 or p >= 1:
            return 0.0
        se = (p * (1 - p) / n) ** 0.5
        return abs(self.realised_rate - p) / se if se > 0 else 0.0

    def render(self) -> str:
        lines = [
            f"NOISE REPORT  process={self.process}  seed={self.seed}",
            f"  target   : {self.target_rate:.4f}",
            f"  realised : {self.realised_rate:.4f}   ({self.n_flipped:,} / {self.n_samples:,})",
            f"  deviation: {self.deviation_sigmas:.2f} binomial SE",
            f"  tolerance: +/- {self.tolerance:.4f}",
        ]
        for c, r in sorted(self.per_class_realised.items()):
            lines.append(f"  class {c}  realised flip rate: {r:.4f}")
        if self.per_client:
            lines.append("  per-client:")
            for rec in self.per_client:
                lines.append(
                    f"    client {rec['client_id']:>3}  n={rec['n']:>6,}  "
                    f"target={rec['target']:.3f}  realised={rec['realised']:.4f}"
                )
        lines.append(f"STATUS: {self.status}")
        return "\n".join(lines)


def _flip(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[mask] = 1 - out[mask]
    return out


class ProtectedLabels(np.ndarray):
    """Label vector the noise injectors must refuse.

    A boolean argument is easy to drop during a refactor. Tagging the array
    itself means `apply_noise(y_test)` raises even if the caller forgets
    `protected=True`.
    """

    def __new__(cls, input_array):
        obj = np.array(input_array, dtype=np.int8, copy=True).view(cls)
        obj.setflags(write=False)
        return obj

    @property
    def protected(self) -> bool:
        return True


def mark_protected(labels) -> ProtectedLabels:
    """Copy `labels` into a write-protected array the injectors will refuse."""
    return ProtectedLabels(labels)


def is_protected(labels, protected: bool = False) -> bool:
    return bool(protected) or bool(getattr(labels, "protected", False))


def _refuse_protected(labels, protected: bool) -> None:
    if is_protected(labels, protected):
        raise NoiseValidationError(
            "refusing to inject noise into a protected label array "
            "(validation and test labels must never be noised - RESEARCH.md §4.2)"
        )


def apply_noise(
    labels: np.ndarray,
    process: str = "symmetric",
    rate: float = 0.0,
    class_rates: dict[int, float] | None = None,
    seed: int = 0,
    tolerance: float | None = None,
    tolerance_sigmas: float = 4.0,
    protected: bool = False,
) -> tuple[np.ndarray, NoiseReport]:
    """Return (noisy_labels, report). Does not mutate the input array.

    `protected=True` is a tripwire for the clean test set. The runner also
    wraps val/test labels with `mark_protected` so a future refactor cannot
    quietly contaminate them by omitting the flag.
    """
    _refuse_protected(labels, protected)
    y = np.asarray(labels, dtype=np.int8)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0,1], got {rate}")

    rng = np.random.default_rng(seed)
    n = len(y)

    if process == "symmetric":
        mask = rng.random(n) < rate
    elif process == "class_conditional":
        if not class_rates:
            raise ValueError("class_conditional noise requires class_rates")
        mask = np.zeros(n, dtype=bool)
        draws = rng.random(n)
        for cls, r in class_rates.items():
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"class rate for {cls} out of range: {r}")
            mask |= (y == cls) & (draws < r)
    else:
        raise ValueError(f"unknown noise process: {process!r}")

    noisy = _flip(y, mask)
    n_flipped = int(mask.sum())
    realised = n_flipped / n if n else 0.0

    per_class = {}
    for cls in np.unique(y):
        sel = y == cls
        per_class[int(cls)] = float(mask[sel].mean()) if sel.any() else 0.0

    # For class_conditional the headline target is the expected overall rate.
    if process == "class_conditional":
        expected = float(
            sum((y == c).sum() * r for c, r in class_rates.items()) / n
        ) if n else 0.0
    else:
        expected = rate

    tol = (
        effective_tolerance(expected, n, sigmas=tolerance_sigmas)
        if tolerance is None
        else float(tolerance)
    )
    rep = NoiseReport(
        process=process,
        target_rate=expected,
        realised_rate=realised,
        n_samples=n,
        n_flipped=n_flipped,
        seed=seed,
        tolerance=tol,
        per_class_realised=per_class,
    )
    rep.status = "PASS" if abs(realised - expected) <= tol else (
        f"BLOCKED: realised {realised:.4f} deviates from target {expected:.4f} "
        f"by {rep.deviation_sigmas:.2f} SE, beyond tolerance {tol:.4f}"
    )
    return noisy, rep


def apply_client_noise(
    labels: np.ndarray,
    client_indices: list[np.ndarray],
    client_rates: list[float],
    seed: int = 0,
    tolerance: float | None = None,
    aggregate_tolerance: float | None = None,
    tolerance_sigmas: float = 4.0,
    protected: bool = False,
) -> tuple[np.ndarray, NoiseReport]:
    """Heterogeneous noise: each client gets its own rate.

    Per-client tolerance is checked but does NOT block on its own, because a
    small shard cannot realise a target rate precisely - with 50 samples and a
    10% target, the realised rate moves in 2% steps. Only the aggregate rate
    blocks. Per-client deviations are reported so they can be inspected.
    """
    _refuse_protected(labels, protected)
    if len(client_indices) != len(client_rates):
        raise ValueError("client_indices and client_rates must be the same length")

    y = np.asarray(labels, dtype=np.int8)
    out = y.copy()
    seq = np.random.SeedSequence(seed)
    child_seeds = seq.spawn(len(client_indices))

    records, total_flipped, total_n = [], 0, 0
    for cid, (idx, rate, cs) in enumerate(zip(client_indices, client_rates, child_seeds)):
        rng = np.random.default_rng(cs)
        idx = np.asarray(idx, dtype=np.int64)
        mask = rng.random(len(idx)) < rate
        out[idx[mask]] = 1 - out[idx[mask]]
        realised = float(mask.mean()) if len(idx) else 0.0
        per_client_tol = (
            effective_tolerance(rate, len(idx), sigmas=tolerance_sigmas)
            if tolerance is None else float(tolerance)
        )
        records.append(
            {
                "client_id": cid,
                "n": int(len(idx)),
                "target": float(rate),
                "realised": realised,
                "n_flipped": int(mask.sum()),
                "tolerance": per_client_tol,
                "within_tolerance": bool(abs(realised - rate) <= per_client_tol),
            }
        )
        total_flipped += int(mask.sum())
        total_n += len(idx)

    expected = (
        float(sum(len(i) * r for i, r in zip(client_indices, client_rates)) / total_n)
        if total_n
        else 0.0
    )
    realised_overall = total_flipped / total_n if total_n else 0.0

    agg_tol = (
        effective_tolerance(expected, total_n, sigmas=tolerance_sigmas)
        if aggregate_tolerance is None
        else float(aggregate_tolerance)
    )
    rep = NoiseReport(
        process="client_heterogeneous",
        target_rate=expected,
        realised_rate=realised_overall,
        n_samples=total_n,
        n_flipped=total_flipped,
        seed=seed,
        tolerance=agg_tol,
        per_client=records,
    )
    rep.status = (
        "PASS"
        if abs(realised_overall - expected) <= agg_tol
        else f"BLOCKED: aggregate realised {realised_overall:.4f} vs target "
             f"{expected:.4f} ({rep.deviation_sigmas:.2f} SE)"
    )
    return out, rep


def assert_noise_valid(report: NoiseReport) -> None:
    """Hard gate (master spec §27, §64)."""
    if not report.status.startswith("PASS"):
        raise NoiseValidationError(f"{report.status}\n\n{report.render()}")
