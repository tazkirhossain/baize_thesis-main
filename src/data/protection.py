"""Immutable validation/test label arrays; never resample protected labels."""
import numpy as np


class ProtectedLabels(np.ndarray):
    """Evaluation labels marked against training-only transformations.

    A boolean argument is easy to drop during a refactor. Tagging the array
    itself means training-only samplers refuse evaluation arrays even if a caller
    supplies the wrong split name.
    """

    def __new__(cls, input_array):
        obj = np.array(input_array, dtype=np.int8, copy=True).view(cls)
        obj.setflags(write=False)
        return obj

    @property
    def protected(self) -> bool:
        return True


def mark_protected(labels) -> ProtectedLabels:
    """Copy `labels` into a write-protected array training-only transformations must refuse."""
    return ProtectedLabels(labels)

