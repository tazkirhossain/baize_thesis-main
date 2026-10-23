"""Explicit strategy registry: algorithms change, artifact/loading contracts do not."""
from functools import partial

from src.data.iid_partitioning import stratified_iid_indices
from src.data.non_iid_partitioning import dirichlet_indices


STRATEGIES = {
    "iid": (stratified_iid_indices, {}),
    "dirichlet": (dirichlet_indices, {"alpha": 0.5, "stratify_on": "category"}),
}


def resolve_strategy(name, parameters=None):
    if name not in STRATEGIES:
        raise ValueError(f"unknown partition strategy {name!r}; choose {sorted(STRATEGIES)}")
    algorithm, defaults = STRATEGIES[name]
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("strategy parameters must be an object")
    parameters = parameters or {}
    unknown = parameters.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"unknown parameters for {name}: {sorted(unknown)}")
    options = {**defaults, **parameters}
    # Validate strategy options before reading the corpus or creating artifacts.
    if name == "dirichlet":
        dirichlet_indices([0, 1], [0, 1], **options)
    descriptor = {"strategy": "stratified_url_iid" if name == "iid" else "dirichlet_url_non_iid",
                  "name": name, "parameters": options,
                  "strata": ["binary_label", "normalized_original_category"] if name == "iid"
                  else ["binary_label" if options["stratify_on"] == "label" else "normalized_original_category"]}
    return partial(algorithm, **options), descriptor
