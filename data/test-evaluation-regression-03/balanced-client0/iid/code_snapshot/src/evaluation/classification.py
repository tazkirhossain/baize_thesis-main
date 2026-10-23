"""Both-class metrics from additive confusion counts; 0=Non-adult, 1=Adult."""
import numpy as np


def confusion_metrics(matrix):
    cm = np.asarray(matrix)
    if (cm.shape != (2, 2) or not np.isfinite(cm).all() or (cm < 0).any() or
            cm.sum() <= 0 or not np.array_equal(cm, np.floor(cm))):
        raise ValueError("a nonnegative nonempty 2x2 confusion matrix is required")
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    divide = lambda a, b: a/b if b else 0.0
    adult_p, adult_r = divide(tp, tp+fp), divide(tp, tp+fn)
    non_p, non_r = divide(tn, tn+fn), divide(tn, tn+fp)
    adult_f1, non_f1 = divide(2*tp, 2*tp+fp+fn), divide(2*tn, 2*tn+fp+fn)
    return {"adult_precision": adult_p, "adult_recall": adult_r, "adult_f1": adult_f1,
            "non_adult_precision": non_p, "non_adult_recall": non_r, "non_adult_f1": non_f1,
            "macro_f1": (adult_f1+non_f1)/2, "balanced_accuracy": (adult_r+non_r)/2,
            "accuracy": divide(tp+tn, tn+fp+fn+tp), "confusion_matrix": [[tn, fp], [fn, tp]],
            "confusion_matrix_order": ["Non-adult", "Adult"], "n_records": tn+fp+fn+tp,
            "adult_count": tp+fn, "non_adult_count": tn+fp,
            "adult_prevalence": divide(tp+fn, tn+fp+fn+tp)}
