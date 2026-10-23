"""Training-only index samplers shared by client-local neural models and numerical utilities.

Ratios always mean Adult:Non-adult. Sampling never edits original partitions.
Oversampling deliberately repeats TRAINING indices; client ownership stays unique.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib

import numpy as np


METHODS = ("none", "undersampling", "oversampling", "smote", "ensemble_undersampling")


@dataclass(frozen=True)
class BalanceConfig:
    method: str = "undersampling"
    adult_to_non_adult: str = "1:1"
    seed: int = 42
    ensemble_members: int = 5
    k_neighbors: int = 5
    insufficient_minority: str = "adjust"  # adjust k, or skip SMOTE explicitly

    def validate(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        ratio(self.adult_to_non_adult)
        for name in ("seed", "ensemble_members", "k_neighbors"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid {name}")
        if self.seed >= 2**32:
            raise ValueError("seed must fit in an unsigned 32-bit integer")
        if self.insufficient_minority not in ("adjust", "skip"):
            raise ValueError("insufficient_minority must be adjust or skip")
        if self.method == "ensemble_undersampling" and self.ensemble_members < 2:
            raise ValueError("an undersampling ensemble requires at least two independent members")


def ratio(value):
    try:
        a, b = value.split(":")
        a, b = int(a), int(b)
        if not 0 < a <= b:
            raise ValueError()
        return Fraction(a, b)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("ratio must be Adult:Non-adult, e.g. '1:1' or '1:5'") from None


def training_labels(labels, *, split):
    # Check the protection marker BEFORE np.asarray can strip the array subclass.
    if split != "train" or getattr(labels, "protected", False):
        raise ValueError("balancing is permitted on the local training split only")
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("nonempty one-dimensional binary labels are required")
    return y


def class_counts(y):
    return {"Adult": int(np.count_nonzero(y == 1)), "Non-adult": int(np.count_nonzero(y == 0))}


@dataclass(frozen=True)
class IndexSelection:
    indices: np.ndarray
    report: dict

    def apply(self, training_dataset):
        """Model-agnostic view; the original dataset can tokenize URLs on demand."""
        if (getattr(training_dataset,"split","train") != "train" or
                getattr(getattr(training_dataset,"labels",None),"protected",False) or
                len(training_dataset) != sum(self.report["before"].values())):
            raise ValueError("apply this selection to its original training dataset only")
        return IndexedTrainingView(training_dataset, self.indices)


class IndexedTrainingView:
    """Works with sequence-style datasets, including future torch Dataset adapters."""
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices
        if len(indices) and (indices.min() < 0 or indices.max() >= len(dataset)):
            raise ValueError("selection does not match this training dataset")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[int(self.indices[item])]


def selection_report(y, selected_y, cfg, **extra):
    before, after = class_counts(y), class_counts(selected_y)
    return {"method":cfg.method, "split":"train", "config":asdict(cfg),
            "before":before, "after":after,
            "achieved_adult_to_non_adult":after["Adult"] / after["Non-adult"] if after["Non-adult"] else None,
            "status":"applied", "duplicated_examples":0, "synthetic_examples":0,
            "removed_training_examples":0, **extra}


def sample_training_indices(labels, config, *, split):
    config.validate()
    y = training_labels(labels, split=split)
    if config.method not in ("none", "undersampling", "oversampling"):
        raise ValueError("use the dedicated SMOTE or ensemble training API for this method")
    adult, non = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(config.seed)
    r = ratio(config.adult_to_non_adult)
    status = "applied"
    if not len(adult) or not len(non):
        indices = np.arange(len(y))
        status = "skipped_missing_class"
    elif config.method == "undersampling":
        target = len(adult) * r.denominator // r.numerator
        # Retain ALL Adult rows. Never add majority records to fill a target.
        chosen = rng.choice(non, size=min(target, len(non)), replace=False)
        indices = rng.permutation(np.concatenate([adult, chosen]))
        if target >= len(non):
            status = "unchanged_already_at_or_above_target"
    elif config.method == "oversampling":
        target = (len(non) * r.numerator + r.denominator - 1) // r.denominator
        extra = max(0, target - len(adult))
        duplicated = rng.choice(adult, size=extra, replace=True)
        indices = rng.permutation(np.concatenate([np.arange(len(y)), duplicated]))
        if not extra:
            status = "unchanged_already_at_or_above_target"
    else:
        indices = np.arange(len(y))
        status = "baseline_unbalanced"
    indices = np.asarray(indices, dtype=np.int64)
    duplicates = len(indices) - len(np.unique(indices))
    report = selection_report(y, y[indices], config, status=status,
        duplicated_examples=duplicates, removed_training_examples=len(y)-len(np.unique(indices)),
        selection_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest())
    indices.setflags(write=False)
    return IndexSelection(indices, report)


def ensemble_training_indices(labels, config, *, split):
    from dataclasses import replace
    config.validate()
    training_labels(labels, split=split)
    if config.method != "ensemble_undersampling":
        raise ValueError("ensemble_undersampling configuration required")
    seeds = np.random.SeedSequence(config.seed).spawn(config.ensemble_members)
    selections = []
    for member, child in enumerate(seeds):
        seed = int(child.generate_state(1)[0])
        selected = sample_training_indices(labels, replace(config, method="undersampling", seed=seed), split="train")
        selected.report.update({"method":"ensemble_undersampling", "member":member+1,
                                "member_seed":seed, "ensemble_seed":config.seed})
        selections.append(selected)
    return selections


def print_balance_report(report, *, prefix=""):
    before, after = report["before"], report["after"]
    print(f"{prefix}{report['method']} [{report['status']}] seed={report['config']['seed']} "
          f"target Adult:Non-adult={report['config']['adult_to_non_adult']}", flush=True)
    print(f"  BEFORE: Adult={before['Adult']:,}, Non-adult={before['Non-adult']:,}", flush=True)
    print(f"  AFTER:  Adult={after['Adult']:,}, Non-adult={after['Non-adult']:,}; "
          f"duplicated={report['duplicated_examples']:,}, synthetic={report['synthetic_examples']:,}, "
          f"unused originals={report['removed_training_examples']:,}", flush=True)
