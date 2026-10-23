"""Aggregate confusion counts and loss sums, without collecting predictions."""
import math

import numpy as np
from src.evaluation.balancing_comparison import pooled_confusion_metrics


def aggregate_validation(results):
    matrix = np.zeros((2, 2), dtype=np.int64)
    loss_sum, total = 0.0, 0
    for count, metrics in results:
        if type(count) is not int or count < 1:
            raise ValueError("invalid evaluation count")
        cells = [metrics[key] for key in ("tn", "fp", "fn", "tp")]
        if any(type(cell) is not int or cell < 0 for cell in cells) or sum(cells) != count:
            raise ValueError("invalid evaluation confusion counts")
        if not math.isfinite(metrics["loss_sum"]) or metrics["loss_sum"] < 0:
            raise ValueError("invalid evaluation loss")
        matrix += np.array(cells).reshape(2, 2)
        total += count
        loss_sum += metrics["loss_sum"]
    aggregate = pooled_confusion_metrics(matrix)
    aggregate["loss"] = loss_sum / total
    return aggregate
