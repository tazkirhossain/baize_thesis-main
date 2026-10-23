"""Read a client's own immutable records without building hashed text features."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.data.protection import mark_protected
from src.data.preprocessing import _file_hash


@dataclass(frozen=True)
class LocalSplit:
    split: str
    urls: tuple
    labels: np.ndarray
    source_rows: np.ndarray
    sha256: str

    def __len__(self):
        return len(self.labels)


def read_local_split(client_directory, split):
    if split not in ("train", "val", "test"):
        raise ValueError("unknown local split")
    directory = Path(client_directory) / "splits"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if _file_hash(directory / "preprocessing.json") != manifest["preprocessing_sha256"]:
        raise ValueError("local preprocessing checksum mismatch")
    path = directory / f"{split}.jsonl"
    if _file_hash(path) != manifest["splits"][split]["sha256"]:
        raise ValueError("local split checksum mismatch")
    urls, labels, rows = [], [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            urls.append(row["url"])
            labels.append(row["label"])
            rows.append(row["source_row"])
    y = np.asarray(labels)
    if not np.isin(y, [0, 1]).all() or len(y) != manifest["splits"][split]["n_records"]:
        raise ValueError("invalid local labels/counts")
    y = y.astype(np.int8)
    if split != "train":
        y = mark_protected(y)
    y.setflags(write=False)
    rows = np.asarray(rows, dtype=np.int64)
    rows.setflags(write=False)
    return LocalSplit(split, tuple(urls), y, rows, manifest["splits"][split]["sha256"])
