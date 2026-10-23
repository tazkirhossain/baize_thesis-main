"""Engineered numerical URL experiment, distinct from DeBERTa/character CNN."""
from dataclasses import dataclass
from urllib.parse import urlsplit
import hashlib

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.balancing import training_labels


FEATURE_NAMES = ("url_length", "hostname_length", "path_length", "dot_count", "digit_count",
    "hyphen_count", "slash_count", "underscore_count", "percent_count", "question_count",
    "equals_count", "ampersand_count", "at_count", "colon_count", "tilde_count",
    "path_depth", "hostname_dot_count", "hostname_digit_count", "path_digit_count")
SCHEMA = "engineered_url_numeric_v1"


def extract_numeric_urls(urls):
    matrix = np.empty((len(urls), len(FEATURE_NAMES)), dtype=np.float64)
    for i, url in enumerate(urls):
        parsed = urlsplit(url if "://" in url else "https://"+url)
        host, path = parsed.hostname or "", parsed.path
        matrix[i] = [len(url), len(host), len(path), url.count("."), sum(c.isdigit() for c in url),
            *[url.count(c) for c in ("-","/","_","%","?","=","&","@",":","~")],
            len([p for p in path.split("/") if p]), host.count("."),
            sum(c.isdigit() for c in host), sum(c.isdigit() for c in path)]
    return matrix


@dataclass(frozen=True)
class NumericalTrainingMatrix:
    X: np.ndarray
    y: np.ndarray
    scaler_fingerprint: str
    schema: str = SCHEMA
    split: str = "train"

    def validate(self):
        training_labels(self.y, split=self.split)
        if (self.schema != SCHEMA or not isinstance(self.X, np.ndarray) or self.X.ndim != 2 or
                self.X.shape != (len(self.y), len(FEATURE_NAMES)) or self.X.dtype.kind != "f" or
                not np.isfinite(self.X).all() or not self.scaler_fingerprint):
            raise ValueError("SMOTE requires scaled engineered numerical URL features, not URL strings or token IDs")


class URLNumericTransformer:
    """Fit once on original LOCAL TRAIN only, then transform unchanged holdouts."""
    def fit_training(self, split):
        training_labels(split.labels, split=split.split)
        raw = extract_numeric_urls(split.urls)
        self.scaler_ = StandardScaler().fit(raw)
        self.fingerprint_ = hashlib.sha256(np.concatenate([
            self.scaler_.mean_, self.scaler_.scale_]).astype("<f8").tobytes()).hexdigest()
        self.training_rows_ = len(split)
        return NumericalTrainingMatrix(self.scaler_.transform(raw), split.labels, self.fingerprint_)

    def transform(self, split):
        if not hasattr(self, "scaler_"):
            raise ValueError("fit on local training before transforming other splits")
        return self.scaler_.transform(extract_numeric_urls(split.urls))

    def report(self):
        return {"schema":SCHEMA, "feature_names":list(FEATURE_NAMES), "scaler":"StandardScaler",
                "fit_split":"train", "n_fit_records":self.training_rows_,
                "mean":self.scaler_.mean_.tolist(), "scale":self.scaler_.scale_.tolist(),
                "fingerprint":self.fingerprint_}
