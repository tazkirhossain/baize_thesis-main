"""Record fit diagnostics without using evaluation data to adjust training."""
import warnings

import numpy as np


def fit_recorded(model, X, y):
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        model.fit(X,y)
    model.fit_audit_ = {"warnings":[{"category":type(m.message).__name__,"message":str(m.message)} for m in messages],
                        "iterations":np.asarray(model.n_iter_).tolist() if hasattr(model,"n_iter_") else None}
    return model
