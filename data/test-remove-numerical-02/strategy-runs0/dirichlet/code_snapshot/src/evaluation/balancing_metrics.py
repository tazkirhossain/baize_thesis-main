"""Adult-positive metrics at a fixed predeclared threshold; no holdout tuning."""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix
from src.evaluation.classification import confusion_metrics


def balancing_metrics(labels, adult_probabilities, *, threshold=0.5):
    y, p = np.asarray(labels), np.asarray(adult_probabilities)
    if (y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isin(y,[0,1]).all() or
            not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not 0<threshold<1):
        raise ValueError("invalid labels, probabilities or fixed threshold")
    predicted = (p >= threshold).astype(int)
    return {**confusion_metrics(confusion_matrix(y,predicted,labels=[0,1])),
        "pr_auc_average_precision":float(average_precision_score(y,p)) if np.any(y==1) else None,
        "confusion_matrix":confusion_matrix(y,predicted,labels=[0,1]).tolist(),
        "confusion_matrix_order":["Non-adult","Adult"], "threshold":float(threshold),
        "n_records":len(y), "adult_count":int(np.count_nonzero(y==1)),
        "non_adult_count":int(np.count_nonzero(y==0))}


def print_metrics(metrics, *, split):
    print(f"  {split}: Adult precision={metrics['adult_precision']:.4f}, "
          f"recall={metrics['adult_recall']:.4f}, F1={metrics['adult_f1']:.4f}, "
          f"Non-adult precision={metrics['non_adult_precision']:.4f}, "
          f"recall={metrics['non_adult_recall']:.4f}, F1={metrics['non_adult_f1']:.4f}, "
          f"macro-F1={metrics['macro_f1']:.4f}, balanced accuracy={metrics['balanced_accuracy']:.4f}, "
          f"accuracy={metrics['accuracy']:.4f}; confusion={metrics['confusion_matrix']}",flush=True)
