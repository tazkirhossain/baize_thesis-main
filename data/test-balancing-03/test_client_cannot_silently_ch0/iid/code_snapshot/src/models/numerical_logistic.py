"""Unweighted numerical classifiers for controlled balancing experiments."""
from sklearn.linear_model import LogisticRegression, SGDClassifier


def numerical_classifier(kind="sgd_logistic", *, seed=42, parameters=None):
    parameters = dict(parameters or {})
    allowed = {"sgd_logistic":{"alpha", "max_iter", "tol"},
               "logistic_regression":{"C", "max_iter", "tol"}}
    if kind not in allowed or parameters.keys() - allowed[kind]:
        raise ValueError("unknown numerical classifier or parameters")
    if kind == "sgd_logistic":
        defaults = {"alpha":0.0001, "max_iter":30, "tol":0.001}
        return SGDClassifier(loss="log_loss", random_state=seed, class_weight=None,
            early_stopping=False, **{**defaults, **parameters})
    return LogisticRegression(random_state=seed, class_weight=None,
        **{"max_iter":300, **parameters})
