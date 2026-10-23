"""Training-budget accounting.

RESEARCH.md §4.5 requires centralized vs federated comparisons at comparable
total gradient steps. The H1 write-up called the E2 run "matched budget" while
FedAvg saw `rounds * local_epochs` passes per example against
`centralized_epochs` for the baseline (about 5x more local SGD). Recording the
figure on every run makes that mismatch visible instead of easy to miss.
"""

from __future__ import annotations


def passes_per_example(
    *,
    centralized_epochs: int | None = None,
    rounds: int | None = None,
    local_epochs: int = 1,
    participation: float = 1.0,
) -> float:
    """How many times each training example is seen, in expectation.

    Centralized: one pass per epoch.
    Federated with full coverage of the training set each round: one pass per
    (round × local epoch), scaled by the participation fraction.
    """
    if centralized_epochs is not None and rounds is not None:
        raise ValueError("provide either centralized_epochs or rounds, not both")
    if centralized_epochs is not None:
        if centralized_epochs < 0:
            raise ValueError(f"centralized_epochs must be >= 0, got {centralized_epochs}")
        return float(centralized_epochs)
    if rounds is None:
        raise ValueError("provide centralized_epochs or rounds")
    if rounds < 0 or local_epochs < 0:
        raise ValueError("rounds and local_epochs must be >= 0")
    return float(rounds) * float(local_epochs) * float(participation)


def describe_budget(n_train: int, passes: float) -> dict:
    return {
        "n_train": int(n_train),
        "passes_per_example": float(passes),
        "sample_exposures": int(round(n_train * passes)),
        "note": (
            "Matched-budget H1 requires equal passes_per_example for the "
            "centralized and federated arms. Do not describe a comparison as "
            "matched unless those figures are equal."
        ),
    }
