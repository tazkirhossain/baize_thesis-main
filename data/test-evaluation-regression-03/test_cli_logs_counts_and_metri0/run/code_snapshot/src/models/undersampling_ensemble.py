"""Independent models fitted to training-only subsets; average Adult probabilities."""
import numpy as np

from src.data.balancing import ensemble_training_indices, print_balance_report, training_labels
from src.models.training_audit import fit_recorded


class UndersamplingEnsemble:
    """Factory(seed)->independent fit/predict_proba model; not a boosting method.

    Future neural ensemble adapters require their own model factory and result
    family. This class does not equate numerical ensembles with neural ensembles.
    """
    def __init__(self, model_factory, config):
        self.model_factory, self.config = model_factory, config

    def fit_training(self, X, labels, *, split="train", verbose=True):
        y = training_labels(labels, split=split)
        if len(X) != len(y):
            raise ValueError("training features and labels differ in length")
        selections = ensemble_training_indices(labels, self.config, split=split)
        self.models_, self.reports_ = [], []
        for selection in selections:
            if verbose:
                print_balance_report(selection.report, prefix=f"Member {selection.report['member']}: ")
            if selection.report["status"] == "skipped_missing_class":
                raise ValueError("cannot train ensemble without both training classes")
            model = self.model_factory(selection.report["member_seed"])
            if any(model is existing for existing in self.models_):
                raise ValueError("model factory must create independent ensemble members")
            indices = selection.indices
            fit_recorded(model,X[indices],y[indices])
            selection.report["fit_audit"] = model.fit_audit_
            if verbose:
                print(f"  Fit iterations={model.fit_audit_['iterations']}; warnings={len(model.fit_audit_['warnings'])}",flush=True)
            self.models_.append(model)
            self.reports_.append(selection.report)
        return self

    def predict_proba(self, X):
        if not getattr(self, "models_", None):
            raise ValueError("ensemble is not fitted")
        probability = np.zeros(len(X), dtype=np.float64)
        for model in self.models_:
            positive = np.flatnonzero(np.asarray(model.classes_) == 1)
            if len(positive) != 1:
                raise ValueError("ensemble member lacks Adult probability column")
            probability += model.predict_proba(X)[:,positive[0]]
        probability /= len(self.models_)
        return np.column_stack([1-probability, probability])

    @property
    def classes_(self):
        return np.array([0,1])
