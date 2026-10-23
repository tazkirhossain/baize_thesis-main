"""Five-client URL-level IID baseline from an audited, already-cleaned corpus.

    The only sampling operation is a seeded permutation inside each
    (binary label, normalized source category) stratum. All records are used
    exactly once. Domains may span clients; a shared local split policy keeps
    each domain in the same train/validation/test role across the federation.
"""
from __future__ import annotations

from array import array
from collections import Counter
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from baize.data.dataset import binary_label
from baize.data.loaders import KNOWN_CATEGORIES
from baize.data.preprocessing import (
    PreprocessingError, _file_hash, _json_write, _snapshot_reproduction,
)
from baize.data.url import normalize_url
from baize.provenance import environment_record


N_CLIENTS = 5


def stratified_iid_indices(labels, categories, seed: int = 0, n_clients: int = 5):
    """Partition row indices without replacement with integer-balanced quotas.

    Each binary-class count, total count, and (label,category) stratum count
    differs by at most one between clients. Category marginals also differ by
    at most one when each category maps to a single binary class, as here.
    """
    if n_clients != N_CLIENTS:
        raise ValueError("this IID baseline requires exactly five clients")
    y = np.asarray(labels)
    cats = np.asarray(categories)
    if y.ndim != 1 or cats.ndim != 1 or y.shape != cats.shape:
        raise ValueError("labels and categories must be parallel one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or any(np.count_nonzero(y == c) < 5 for c in (0, 1)):
        raise ValueError("binary labels and at least five records of each class are required")
    rng = np.random.default_rng(seed)
    totals = np.full(5, len(y) // 5, dtype=np.int64)
    totals[rng.permutation(5)[:len(y) % 5]] += 1
    class_sizes = np.array([np.count_nonzero(y == c) for c in (0, 1)])
    quotas = np.repeat((class_sizes // 5)[:, None], 5, axis=1)
    remaining = totals - quotas.sum(axis=0)
    for c in (0, 1):
        order = sorted(rng.permutation(5), key=lambda i: -remaining[i])
        for i in order[:int(class_sizes[c] % 5)]:
            quotas[c, i] += 1
            remaining[i] -= 1
    if np.any(remaining):
        raise RuntimeError("binary quota allocation failed")
    pieces = [[] for _ in range(5)]
    for c in (0, 1):
        groups = [np.flatnonzero((y == c) & (cats == category))
                  for category in np.unique(cats[y == c])]
        # Largest-remainder first, randomly breaking equal-size ties.
        order = rng.permutation(len(groups))
        groups = [groups[i] for i in sorted(order, key=lambda i: -(len(groups[i]) % 5))]
        capacity = quotas[c] - sum(len(group) // 5 for group in groups)
        for group in groups:
            shuffled = rng.permutation(group)
            sizes = np.full(5, len(group) // 5, dtype=np.int64)
            order = sorted(rng.permutation(5), key=lambda i: -capacity[i])
            for i in order[:len(group) % 5]:
                if capacity[i] <= 0:
                    raise RuntimeError("category quota allocation failed")
                sizes[i] += 1
                capacity[i] -= 1
            for i, part in enumerate(np.split(shuffled, np.cumsum(sizes)[:-1])):
                pieces[i].append(part)
        if np.any(capacity):
            raise RuntimeError("unfilled category quota")
    parts = [np.sort(np.concatenate(items)) for items in pieces]
    joined = np.concatenate(parts)
    if not np.array_equal(np.sort(joined), np.arange(len(y))):
        raise RuntimeError("IID partition does not cover each row exactly once")
    return parts


def _read_input(source, report):
    """Validate fingerprints and identities while retaining only compact metadata."""
    source_rows, labels, categories = array("q"), array("b"), array("h")
    names = sorted(KNOWN_CATEGORIES)
    codes = {name: i for i, name in enumerate(names)}
    canonicals = set()
    files = []
    limit = report["ingestion"]["input_rows"] + int(report["input_schema"]["header_detected"])
    for client in sorted(report["clients"], key=lambda c: c["client_id"]):
        path = source / f"client_{client['client_id']:03d}" / "data.jsonl"
        files.append(path)
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    raise PreprocessingError("invalid JSON in prepared source") from None
                required = {"source_row", "record_id", "original_category", "category",
                            "label", "target", "url", "domain"}
                if (not isinstance(row, dict) or not required.issubset(row) or
                        any(not isinstance(row[k], str) for k in
                            ("record_id", "original_category", "category", "target", "url", "domain"))):
                    raise PreprocessingError("prepared source is missing required identity fields")
                try:
                    label = binary_label(row["label"])
                except (ValueError, TypeError):
                    raise PreprocessingError("invalid binary label in prepared source") from None
                category = row["original_category"].strip().lower()
                number = row["source_row"]
                normalized = normalize_url(row["url"])
                if (type(number) is not int or not 1 <= number <= limit or
                        not isinstance(row["record_id"], str) or not row["record_id"].strip() or
                        category not in codes or row["category"] != category or
                        label != int(category == "adult") or
                        row["target"] != ("Adult" if label else "Non-adult") or
                        not normalized.is_valid or normalized.raw != row["url"] or
                        normalized.registrable_domain != row["domain"]):
                    raise PreprocessingError("prepared record violates the audited source schema")
                if normalized.canonical in canonicals:
                    raise PreprocessingError("duplicate canonical URL in prepared source")
                canonicals.add(normalized.canonical)
                source_rows.append(number)
                labels.append(label)
                categories.append(codes[category])
                count += 1
        if digest.hexdigest() != client["dataset"]["sha256"] or count != client["dataset"]["n_records"]:
            raise PreprocessingError("source client file checksum/count mismatch")
    numbers = np.asarray(source_rows, dtype=np.int64)
    order = np.argsort(numbers)
    numbers = numbers[order]
    if len(numbers) != report["retained_records"] or np.any(np.diff(numbers) <= 0):
        raise PreprocessingError("source identity coverage/uniqueness failed")
    return numbers, np.asarray(labels, dtype=np.int8)[order], np.asarray(categories)[order], names, files


def partition_prepared_iid(source: str | Path, output: str | Path, *, seed=0,
                           split_seed=0, n_clients=5):
    """Repartition the clean corpus; preserve the existing cleaning audit verbatim.

    Creates unsplit client-owned files. Each client later calls split_client;
    the shared domain rule is carried in its own local configuration.
    """
    if n_clients != 5:
        raise ValueError("this IID baseline requires exactly five clients")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite IID allocation: {output}")
    source_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    settings = json.loads((source / "preprocessing.json").read_text(encoding="utf-8"))
    if source_report.get("status") != "PASS" or settings.get("schema_version") != 3:
        raise PreprocessingError("IID allocation requires a verified schema-v3 clean source")
    print("Validating cleaned source identities, URL uniqueness and file checksums...", flush=True)
    numbers, labels, categories, names, files = _read_input(source, source_report)
    parts = stratified_iid_indices(labels, categories, seed, n_clients)
    owner = np.full(int(numbers[-1]) + 1, -1, dtype=np.int8)
    for cid, indices in enumerate(parts):
        owner[numbers[indices]] = cid
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preparing-iid-", dir=output.parent) as temporary:
        ready = Path(temporary) / "ready"
        ready.mkdir()
        settings = copy.deepcopy(settings)
        settings["config"]["n_clients"] = 5
        settings["config"]["seed"] = int(seed)
        settings["partitioning"] = {"strategy": "stratified_url_iid", "n_clients": 5,
            "seed": int(seed), "rng": "numpy.PCG64", "replacement": False,
            "strata": ["binary_label", "normalized_original_category"],
            "population_order": "ascending original source_row"}
        settings["split_policy"] = {"type": "shared_domain_hash_v1", "seed": int(split_seed)}
        settings["domain_policy"] = "domains may span clients, but have one shared split role"
        settings["sampling"] = "all retained records exactly once; seeded permutation without replacement"
        clients = []
        with ExitStack() as stack:
            handles = []
            for cid in range(5):
                directory = ready / f"client_{cid:03d}"
                directory.mkdir()
                handles.append(stack.enter_context((directory / "data.jsonl").open("wb")))
            # Merge the pre-sorted input streams: output order is independent of
            # the old ownership assignment and stays reproducible by source_row.
            import heapq
            def records(path):
                with path.open("rb") as stream:
                    for line in stream:
                        yield json.loads(line)["source_row"], line
            written = np.zeros(len(owner), dtype=np.bool_)
            for number, line in heapq.merge(*(records(path) for path in files), key=lambda p: p[0]):
                cid = int(owner[number])
                if cid < 0 or written[number]:
                    raise PreprocessingError("output identity assignment failed")
                handles[cid].write(line)
                written[number] = True
        if int(written.sum()) != len(numbers):
            raise PreprocessingError("IID output record coverage failed")
        for cid, indices in enumerate(parts):
            counts = Counter(int(v) for v in labels[indices])
            category_counts = {names[int(code)]: int(np.count_nonzero(categories[indices] == code))
                               for code in np.unique(categories[indices])}
            info = {"n_records": len(indices), "class_counts": {str(c): counts[c] for c in (0, 1)},
                "positive_rate": counts[1] / len(indices), "category_counts": category_counts,
                "sha256": _file_hash(ready / f"client_{cid:03d}/data.jsonl")}
            _json_write(ready / f"client_{cid:03d}/preprocessing.json", settings)
            client = {"client_id": cid, "state": "allocated_unsplit", "dataset": info,
                "preprocessing_sha256": _file_hash(ready / f"client_{cid:03d}/preprocessing.json")}
            _json_write(ready / f"client_{cid:03d}/manifest.json", client)
            clients.append(client)
        _json_write(ready / "preprocessing.json", settings)
        # Preserve rejected source identities and verify their recorded hashes.
        for name, expected in (("exclusions.jsonl", source_report["excluded_records"]["sha256"]),
                               ("quarantine.jsonl", source_report["quarantine_sha256"])):
            shutil.copyfile(source / name, ready / name)
            if _file_hash(ready / name) != expected:
                raise PreprocessingError("cleaning audit checksum mismatch")
        _json_write(ready / "cleaning_report.json", source_report)
        reproduction = _snapshot_reproduction(ready)
        metrics = {}
        for kind in ("n_records", "adult", "non_adult", *names):
            values = [c["dataset"]["n_records"] if kind == "n_records" else
                      c["dataset"]["class_counts"]["1" if kind == "adult" else "0"]
                      if kind in ("adult", "non_adult") else
                      c["dataset"]["category_counts"].get(kind, 0) for c in clients]
            metrics[kind] = {"min": min(values), "max": max(values), "spread": max(values)-min(values)}
        if any(v["spread"] > 1 for v in metrics.values()):
            raise PreprocessingError("IID quota balance verification failed")
        report = {"status": "PASS", "partitioning": settings["partitioning"],
            "split_policy": settings["split_policy"], "source_sha256": source_report["source_sha256"],
            "prepared_source_report_sha256": _file_hash(source / "report.json"),
            "ingestion": source_report["ingestion"], "input_schema": source_report["input_schema"],
            "retained_records": len(numbers), "clients": clients, "distribution_spreads": metrics,
            "environment": environment_record(), "reproduction": reproduction,
            "excluded_records": source_report["excluded_records"],
            "quarantine_sha256": source_report["quarantine_sha256"],
            "checks": {"exactly_five_clients": True, "sampling_without_replacement": True,
                "complete_record_coverage": True, "unique_source_rows": True,
                "unique_canonical_urls": True, "binary_category_size_spreads_at_most_one": True},
            "notes": ["No records were added, removed, relabelled or resampled.",
                "Domains can span clients. Every client must use shared_domain_hash_v1 for local splitting.",
                "Local split fractions are approximate because domains stay in one shared role."]}
        for path, client in zip(files, sorted(source_report["clients"], key=lambda c:c["client_id"])):
            if _file_hash(path) != client["dataset"]["sha256"]:
                raise PreprocessingError("prepared source changed during IID allocation")
        _json_write(ready / "report.json", report)
        ready.rename(output)
    return report
