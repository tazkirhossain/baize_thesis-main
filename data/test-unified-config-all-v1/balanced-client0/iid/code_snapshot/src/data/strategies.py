"""Explicit strategy registry: algorithms change, artifact/loading contracts do not."""
from functools import partial
from dataclasses import dataclass
from typing import Callable

from src.data.iid_partitioning import stratified_iid_indices
from src.data.non_iid_partitioning import dirichlet_indices, validate_dirichlet_parameters
from src.data.planned_partitioning import planned_indices, validate_label_plan, validate_category_plan


@dataclass(frozen=True)
class Strategy:
    allocate: Callable
    defaults: dict
    validate: Callable
    describe: Callable


STRATEGIES = {
    "iid": Strategy(stratified_iid_indices, {}, lambda: None,
        lambda options: {"strategy":"stratified_url_iid", "balanced_margins":True,
                         "strata":["binary_label", "normalized_original_category"]}),
    "dirichlet": Strategy(dirichlet_indices, {"alpha":0.5, "stratify_on":"category"},
        validate_dirichlet_parameters,
        lambda options: {"strategy":"dirichlet_url_non_iid", "balanced_margins":False,
            "strata":["binary_label" if options["stratify_on"] == "label" else "normalized_original_category"]}),
    "label_skew": Strategy(partial(planned_indices, axis="label"), {"allocation_plan":None},
        validate_label_plan, lambda options: {"strategy":"planned_label_skew", "balanced_margins":False,
                                            "strata":["binary_label"]}),
    "category_skew": Strategy(partial(planned_indices, axis="category"), {"allocation_plan":None},
        validate_category_plan, lambda options: {"strategy":"planned_category_skew", "balanced_margins":False,
                                               "strata":["normalized_original_category"]}),
}


def resolve_strategy(name, parameters=None):
    if name not in STRATEGIES:
        raise ValueError(f"unknown partition strategy {name!r}; choose {sorted(STRATEGIES)}")
    spec = STRATEGIES[name]
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("strategy parameters must be an object")
    parameters = parameters or {}
    unknown = parameters.keys() - spec.defaults.keys()
    if unknown:
        raise ValueError(f"unknown parameters for {name}: {sorted(unknown)}")
    options = {**spec.defaults, **parameters}
    # Validate strategy options before reading the corpus or creating artifacts.
    spec.validate(**options)
    descriptor = {**spec.describe(options), "name":name, "parameters":options}
    return partial(spec.allocate, **options), descriptor
