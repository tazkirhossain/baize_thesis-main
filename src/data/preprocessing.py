"""Offline CSV preparation for client-owned, domain-disjoint FL datasets.

This is a simulation provisioning tool, not a server-side training data store.
No network access, LLM calls, learned vocabulary, resampling, or label inference.
SQLite keeps corpus-sized deduplication off the Python heap. Publication of an
output directory is atomic: failed integrity gates never produce a ready dataset.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import tempfile
import shutil
import importlib.metadata
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.data.features import build_vectorizer
from src.data.schema import binary_label
from src.data.schema import KNOWN_CATEGORIES
from src.data.protection import mark_protected
from src.data.url import normalize_url
from src.provenance import environment_record
from packaging.requirements import Requirement


class PreprocessingError(ValueError):
    """The corpus cannot satisfy the declared client/split integrity policy."""


@dataclass(frozen=True)
class PreprocessingConfig:
    n_clients: int = 5
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 0
    n_features: int = 2**16
    ngram_range: tuple[int, int] = (3, 5)
    min_per_class: int = 1
    header: str = "auto"  # auto | present | absent

    def validate(self):
        if self.header not in ("auto", "present", "absent"):
            raise PreprocessingError("header must be auto, present or absent")
        if self.n_clients < 1 or self.min_per_class < 1:
            raise PreprocessingError("n_clients and min_per_class must be positive")
        if (len(self.fractions) != 3 or
                any(not math.isfinite(f) or f <= 0 for f in self.fractions) or
                not math.isclose(sum(self.fractions), 1.0, abs_tol=1e-9)):
            raise PreprocessingError("three positive finite fractions must sum to 1")
        if self.n_features < 1 or not (1 <= self.ngram_range[0] <= self.ngram_range[1]):
            raise PreprocessingError("invalid hashing feature configuration")


SPLITS = ("train", "val", "test")


def _json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_directory(ready, output):
    """Atomically publish, tolerating brief Windows scanner/indexer file locks.

    Never overwrite an output or fall back to non-atomic copying. Only Windows
    sharing/access errors are retried; permanent failures still propagate.
    """
    delays = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0)
    for attempt in range(len(delays) + 1):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite prepared output: {output}")
        try:
            ready.rename(output)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def _allocate(groups, fractions, seed):
    """Assign whole domains, minimizing class-count target error.

    Positive and negative counts are equally weighted relative to their totals.
    Both client ownership and local split allocation use this same rule. Large
    indivisible hosting domains may force unequal sizes; reports expose that.
    """
    groups = sorted(groups, key=lambda g: hashlib.sha256(
        f"{seed}:{g[0]}".encode()).digest())
    groups.sort(key=lambda g: max(g[1], g[2]), reverse=True)
    totals = [sum(g[c + 1] for g in groups) for c in range(2)]
    targets = [[f * totals[c] for c in range(2)] for f in fractions]
    if min(totals, default=0) == 0:
        raise PreprocessingError("both classes are required before allocation")
    loads = [[0, 0] for _ in fractions]
    assigned = {}
    for domain, neg, pos in groups:
        counts = (neg, pos)
        scores = [sum((2 * loads[k][c] * counts[c] + counts[c] ** 2)
                      / targets[k][c] - 2 * counts[c] for c in range(2))
                  for k in range(len(fractions))]
        slot = min(range(len(scores)), key=lambda k: scores[k])
        assigned[domain] = slot
        for c in range(2):
            loads[slot][c] += counts[c]
    return assigned


def _ingest(source, db, quarantine, exclusions, header_mode):
    db.execute("""CREATE TABLE records (
        canonical TEXT PRIMARY KEY, url TEXT, domain TEXT, category TEXT,
        label INTEGER, source_row INTEGER, occurrences INTEGER, label_mask INTEGER,
        record_id TEXT, original_category TEXT
    )""")
    db.execute("""CREATE TABLE source_records (
        canonical TEXT, source_row INTEGER PRIMARY KEY, record_id TEXT,
        original_category TEXT, label INTEGER
    )""")
    counters = Counter()
    categories = Counter()
    original_categories = Counter()
    missing = Counter({"id": 0, "url": 0, "label": 0})
    reason_counts = Counter()
    columns = ["id", "url", "label"]
    schema = {"header_mode": header_mode, "header_detected": False, "columns": columns}
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for row_number, row in enumerate(csv.reader(stream, strict=True), 1):
            if row_number == 1:
                names = [s.strip().lower() for s in row]
                looks_like_header = any(s in {"id", "url", "label", "record_id", "category"}
                                        for s in names)
                if header_mode == "present" or (header_mode == "auto" and looks_like_header):
                    if len(names) != 3 or set(names) != {"id", "url", "label"}:
                        raise PreprocessingError("CSV header must contain exactly id,url,label, without duplicates")
                    columns = names
                    schema.update(header_detected=True, columns=columns)
                    counters["header_rows"] += 1
                    continue
            counters["input_rows"] += 1
            fields = {name: row[i] if i < len(row) else None for i, name in enumerate(columns)}
            for column, value in fields.items():
                if value is None or not value.strip():
                    missing[column] += 1
            record_id, raw_url, original_category = (fields[k] for k in ("id", "url", "label"))
            category = (original_category or "").strip().lower()
            if original_category is not None and original_category.strip():
                original_categories[original_category] += 1
            reasons = []
            if len(row) != 3:
                reasons.append("malformed_columns")
            if record_id is None or not record_id.strip():
                reasons.append("missing_id")
            if raw_url is None or not raw_url.strip():
                reasons.append("missing_url")
            if not category:
                reasons.append("unlabelled")
            normalized = normalize_url(raw_url) if raw_url and raw_url.strip() else None
            if normalized is not None and not normalized.is_valid:
                reasons.append("invalid_url:" + normalized.reason)
            if category and category not in KNOWN_CATEGORIES:
                reasons.append("unknown_category")
            if reasons:
                counters[reasons[0]] += 1
                reason_counts.update(reasons)
                # Invalid originals and unknown free-text labels are never echoed.
                record = {"source_row": row_number, "record_id": record_id,
                          "original_category": original_category,
                          "reason": reasons[0], "reasons": reasons}
                if normalized is not None and normalized.is_valid:
                    record["url"] = normalized.raw
                    record["domain"] = normalized.registrable_domain
                line = json.dumps(record, ensure_ascii=True) + "\n"
                quarantine.write(line)
                exclusions.write(line)
                continue
            label = int(category == "adult")
            categories[category] += 1
            counters["accepted_before_dedup"] += 1
            db.execute("""INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(canonical) DO UPDATE SET
                  occurrences=occurrences+1, label_mask=label_mask | excluded.label_mask
                """, (normalized.canonical, normalized.raw, normalized.registrable_domain,
                      category, label, row_number, 1 << label, record_id, original_category))
            db.execute("INSERT INTO source_records VALUES (?, ?, ?, ?, ?)",
                       (normalized.canonical, row_number, record_id, original_category, label))
            if row_number % 50_000 == 0:
                db.commit()
    db.commit()
    if not counters["input_rows"]:
        raise PreprocessingError("CSV contains no data records")
    return counters, categories, original_categories, missing, reason_counts, schema


def _snapshot_reproduction(ready):
    """Archive exact preparation sources and an installed dependency closure.

    This records what actually ran even in an exported checkout without Git.
    No environment variables, credentials or pip configuration are inspected.
    """
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src").rglob("*.py"))
    paths += [root / name for name in ("config/five_client_iid.json", "config/five_client_non_iid.json",
              "config/five_client_label_skew.json", "config/five_client_category_skew.json",
              "config/five_client_dirichlet.json")]
    paths += [root / f"config/balancing/{method}.json" for method in
              ("undersampling", "oversampling", "smote", "ensemble_undersampling")]
    paths += [root / name for name in ("scripts/prepare_clients.py", "scripts/split_client.py",
                                      "scripts/check_repository_safety.py", "scripts/partition_iid.py",
                                      "scripts/partition_clients.py", "scripts/compare_partitions.py",
                                      "scripts/create_category_skew_config.py", "scripts/verify_partitions.py",
                                      "scripts/prepare_deberta.py", "config/models/deberta_v3_base.json",
                                      "scripts/enrich_client.py", "config/website_collection.json",
                                      "scripts/reuse_enrichment.py",
                                      ".gitignore",
                                      "requirements.txt", "pyproject.toml", "docs/PREPROCESSING.md")]
    hashes = {}
    for path in paths:
        relative = path.relative_to(root)
        destination = ready / "code_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[relative.as_posix()] = _file_hash(destination)
    pending = ["numpy", "scipy", "scikit-learn", "tldextract", "packaging"]
    try:
        importlib.metadata.version("pytest")
        pending.append("pytest")
    except importlib.metadata.PackageNotFoundError:
        pass  # pytest is optional in a production installation.
    installed = {}
    while pending:
        distribution = importlib.metadata.distribution(pending.pop())
        name = distribution.metadata["Name"]
        if name.lower() in installed:
            continue
        installed[name.lower()] = (name, distribution.version)
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    (ready / "runtime-requirements.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in sorted(installed.values())) + "\n",
        encoding="utf-8")
    return {"source_files_sha256": hashes,
            "source_tree_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "runtime_requirements_sha256": _file_hash(ready / "runtime-requirements.txt"),
            "cold_environment_reproduction_verified": False}


def prepare_clients(source: str | Path, output: str | Path,
                    config: PreprocessingConfig | None = None) -> dict:
    """Create immutable per-client JSONL datasets and verified manifests.

    Output must not exist. The source is never modified. Missing/unknown labels
    are quarantined, never imputed. Every retained domain has exactly one client
    owner. No train/validation/test split or feature extraction occurs here.
    Each client later calls split_client on its own directory.
    """
    cfg = config or PreprocessingConfig()
    cfg.validate()
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = _file_hash(source)
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=output.parent) as temporary:
        staging = Path(temporary)
        ready = staging / "ready"
        ready.mkdir()
        db = sqlite3.connect(staging / "dedup.sqlite")
        try:
            with (ready / "quarantine.jsonl").open("w", encoding="utf-8") as quarantine, \
                    (ready / "exclusions.jsonl").open("w", encoding="utf-8") as exclusions:
                counters, categories, original_categories, missing, reasons, schema = _ingest(
                    source, db, quarantine, exclusions, cfg.header)
                conflicts = db.execute("SELECT COUNT(*), COALESCE(SUM(occurrences),0) "
                                       "FROM records WHERE label_mask=3").fetchone()
                for row in db.execute("""SELECT s.source_row,s.record_id,s.original_category,
                        s.label,r.url,r.domain,r.label_mask,r.source_row,r.record_id
                        FROM source_records s JOIN records r ON s.canonical=r.canonical
                        WHERE r.label_mask=3 OR s.source_row<>r.source_row
                        ORDER BY s.source_row"""):
                    source_row, record_id, original_category, label, url, domain, mask, kept_row, kept_id = row
                    reason = "label_conflict" if mask == 3 else "duplicate_url"
                    record = {"source_row": source_row, "record_id": record_id,
                        "original_category": original_category, "label": label,
                        "target": "Adult" if label else "Non-adult", "url": url,
                        "domain": domain, "reason": reason, "reasons": [reason]}
                    if reason == "duplicate_url":
                        record.update(kept_source_row=kept_row, kept_record_id=kept_id)
                    line = json.dumps(record, ensure_ascii=True) + "\n"
                    exclusions.write(line)
                    if reason == "label_conflict":
                        quarantine.write(line)
            db.execute("DELETE FROM records WHERE label_mask=3")
            db.execute("CREATE INDEX record_domains ON records(domain)")
            groups = db.execute("SELECT domain, SUM(label=0), SUM(label=1) "
                                "FROM records GROUP BY domain").fetchall()
            if len(groups) < cfg.n_clients * 3:
                raise PreprocessingError("too few independent domains for client-local splits")
            owners = _allocate(groups, [1 / cfg.n_clients] * cfg.n_clients, cfg.seed)
            db.execute("CREATE TABLE assignments (domain TEXT PRIMARY KEY, client INTEGER)")
            clients = []
            db.executemany("INSERT INTO assignments VALUES (?, ?)", owners.items())
            db.commit()
            del groups, owners

            for client_id in range(cfg.n_clients):
                client_dir = ready / f"client_{client_id:03d}"
                client_dir.mkdir()
                file = client_dir / "data.jsonl"
                labels, category_counts, domains = Counter(), Counter(), set()
                with file.open("w", encoding="utf-8", newline="\n") as handle:
                    rows = db.execute("""SELECT r.url,r.domain,r.category,r.label,r.source_row,
                            r.record_id,r.original_category
                        FROM records r JOIN assignments a ON r.domain=a.domain
                        WHERE a.client=? ORDER BY r.source_row""", (client_id,))
                    for url, domain, category, label, source_row, record_id, original_category in rows:
                        handle.write(json.dumps({"url": url, "domain": domain,
                            "category": category, "label": label, "source_row": source_row,
                            "record_id": record_id, "original_category": original_category,
                            "target": "Adult" if label else "Non-adult"},
                            ensure_ascii=True, sort_keys=True) + "\n")
                        labels[label] += 1
                        category_counts[category] += 1
                        domains.add(domain)
                if any(labels[c] < 3 * cfg.min_per_class for c in (0, 1)):
                    raise PreprocessingError(f"client {client_id}: insufficient class coverage")
                dataset = {"n_records": sum(labels.values()),
                    "class_counts": {str(c): labels[c] for c in (0, 1)},
                    "positive_rate": labels[1] / sum(labels.values()),
                    "n_domains": len(domains), "category_counts": dict(category_counts),
                    "sha256": _file_hash(file)}
                client_report = {"client_id": client_id, "state": "allocated_unsplit", "dataset": dataset}
                _json_write(client_dir / "manifest.json", client_report)
                clients.append(client_report)

            retained = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            if db.execute("SELECT COUNT(*) FROM assignments").fetchone()[0] != db.execute(
                    "SELECT COUNT(DISTINCT domain) FROM records").fetchone()[0]:
                raise PreprocessingError("domain ownership coverage failed")
            if sum(c["dataset"]["n_records"] for c in clients) != retained:
                raise PreprocessingError("record coverage failed")
            rejected = counters["input_rows"] - counters["accepted_before_dedup"]
            duplicate_rows = counters["accepted_before_dedup"] - conflicts[1] - retained
            if counters["input_rows"] != retained + rejected + conflicts[1] + duplicate_rows:
                raise PreprocessingError("source row accounting failed")
            settings = {"schema_version": 3, "config": asdict(cfg),
                "label_mapping": {"Adult": 1, "Non-adult": 0},
                "target_policy": "source category Adult -> Adult; 14 other known categories -> Non-adult",
                "features": {"kind": "hashing", "analyzer": "char_wb", "lowercase": True,
                    "n_features": cfg.n_features, "ngram_range": list(cfg.ngram_range),
                    "norm": "l2", "alternate_sign": False},
                "scaling": "per-record L2 normalization; no fitted corpus statistics",
                "sampling": "all valid unique labelled records; no balancing or resampling",
                "evaluation": "client-owned validation and test; no server dataset",
                "domain_policy": "one registrable domain per client; local domain-disjoint splits"}
            _json_write(ready / "preprocessing.json", settings)
            for client_id in range(cfg.n_clients):
                _json_write(ready / f"client_{client_id:03d}" / "preprocessing.json", settings)
            reproduction = _snapshot_reproduction(ready)
            excluded_counts = Counter()
            excluded_rows = set()
            with (ready / "exclusions.jsonl").open(encoding="utf-8") as ledger:
                for line in ledger:
                    entry = json.loads(line)
                    if entry["source_row"] in excluded_rows:
                        raise PreprocessingError("duplicate source row in exclusion ledger")
                    excluded_rows.add(entry["source_row"])
                    excluded_counts[entry["reason"]] += 1
            if len(excluded_rows) != rejected + duplicate_rows + conflicts[1]:
                raise PreprocessingError("exclusion ledger accounting failed")
            report = {"status": "PASS", "source_sha256": source_hash,
                "environment": environment_record(), "ingestion": dict(counters),
                "reproduction": reproduction, "input_schema": schema,
                "missing_values": dict(missing), "rejection_reason_counts": dict(reasons),
                "raw_source_category_counts": dict(original_categories),
                "accepted_category_counts_before_dedup": dict(categories), "rejected_rows": rejected,
                "duplicate_rows_removed": duplicate_rows,
                "conflicting_urls_removed": conflicts[0], "conflicting_source_rows": conflicts[1],
                "excluded_records": {"n_records": len(excluded_rows), "reason_counts": dict(excluded_counts),
                    "file": "exclusions.jsonl", "sha256": _file_hash(ready / "exclusions.jsonl")},
                "quarantine_sha256": _file_hash(ready / "quarantine.jsonl"),
                "retained_records": retained, "clients": clients,
                "checks": {"row_accounting": True, "all_clients_have_both_classes": True,
                    "unique_domain_ownership": True, "record_coverage": True,
                    "complete_exclusion_ledger": True},
                "notes": ["Class/category proportions and split fractions are approximate because domains are indivisible.",
                    "Quarantine is excluded from all training and evaluation; labels are never guessed.",
                    "No sitemap, robots.txt, metadata fetching or LLM labelling has been performed."]}
            if _file_hash(source) != source_hash:
                raise PreprocessingError("source changed during preparation")
            _json_write(ready / "report.json", report)
        finally:
            db.close()
        _publish_directory(ready, output)
    return report


def split_client(client_directory: str | Path) -> dict:
    """Client-only stage: read this client's unsplit data and publish local splits.

    Needs no global report, source CSV, or other client's directory. Feature
    extraction/normalization occurs only when a split is subsequently loaded.
    Existing splits are never overwritten. Whole domains stay in one split.
    """
    directory = Path(client_directory).resolve()
    settings = json.loads((directory / "preprocessing.json").read_text(encoding="utf-8"))
    cfg = PreprocessingConfig(**{**settings["config"],
        "fractions": tuple(settings["config"]["fractions"]),
        "ngram_range": tuple(settings["config"]["ngram_range"])})
    cfg.validate()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("preprocessing_sha256") and
            _file_hash(directory / "preprocessing.json") != manifest["preprocessing_sha256"]):
        raise PreprocessingError("client preprocessing configuration checksum mismatch")
    source, output = directory / "data.jsonl", directory / "splits"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite local splits: {output}")
    expected = manifest["dataset"]
    if _file_hash(source) != expected["sha256"]:
        raise PreprocessingError("client data checksum mismatch")
    groups = {}
    labels = Counter()
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            label = binary_label(row["label"])
            groups.setdefault(row["domain"], [0, 0])[label] += 1
            labels[label] += 1
    if (sum(labels.values()) != expected["n_records"] or
            {str(c): labels[c] for c in (0, 1)} != expected["class_counts"]):
        raise PreprocessingError("client record/class count mismatch")
    policy = settings.get("split_policy")
    if policy:
        if policy.get("type") != "shared_domain_hash_v1":
            raise PreprocessingError("unknown client split policy")
        from src.data.split_policy import shared_domain_split
        assignments = {domain: shared_domain_split(domain, policy["seed"], cfg.fractions)
                       for domain in groups}
    else:
        assignments = _allocate([(d, counts[0], counts[1]) for d, counts in groups.items()],
                                cfg.fractions, cfg.seed + manifest["client_id"] + 1)
    with tempfile.TemporaryDirectory(prefix=".splitting-", dir=directory) as temporary:
        ready = Path(temporary) / "ready"
        ready.mkdir()
        files = [(ready / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")
                 for split in SPLITS]
        counts = [Counter() for _ in SPLITS]
        categories = [Counter() for _ in SPLITS]
        domains = [set() for _ in SPLITS]
        try:
            with source.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    slot = assignments[row["domain"]]
                    files[slot].write(line)
                    counts[slot][binary_label(row["label"])] += 1
                    categories[slot][row["category"]] += 1
                    domains[slot].add(row["domain"])
        finally:
            for handle in files:
                handle.close()
        reports = {}
        for k, split in enumerate(SPLITS):
            if any(counts[k][c] < cfg.min_per_class for c in (0, 1)):
                raise PreprocessingError(f"client {manifest['client_id']} {split}: "
                    f"needs at least {cfg.min_per_class} records of each class")
            reports[split] = {"n_records": sum(counts[k].values()),
                "class_counts": {str(c): counts[k][c] for c in (0, 1)},
                "positive_rate": counts[k][1] / sum(counts[k].values()),
                "n_domains": len(domains[k]), "category_counts": dict(categories[k]),
                "sha256": _file_hash(ready / f"{split}.jsonl")}
        if sum(r["n_records"] for r in reports.values()) != expected["n_records"]:
            raise PreprocessingError("local split coverage failed")
        if _file_hash(source) != expected["sha256"]:
            raise PreprocessingError("client data changed while splitting")
        result = {"client_id": manifest["client_id"], "state": "split_locally",
                  "source_sha256": expected["sha256"], "splits": reports}
        if manifest.get("preprocessing_sha256"):
            result["preprocessing_sha256"] = manifest["preprocessing_sha256"]
        if policy:
            result["split_policy"] = policy
        _json_write(ready / "manifest.json", result)
        _json_write(ready / "preprocessing.json", settings)
        _publish_directory(ready, output)
    return result


def load_local_split(client_directory: str | Path, split: str):
    """Extract normalized features using only this client's local split files."""
    directory = Path(client_directory) / "splits"
    settings = json.loads((directory / "preprocessing.json").read_text(encoding="utf-8"))
    return _load_split(directory, settings, split)


def load_client_split(root: str | Path, client_id: int, split: str):
    """Client-side feature extraction; validate the artifact before using labels.

    Returns a CSR matrix and labels. Val/test labels are noise-protected. The
    identical stateless transform applies to all clients and splits; no fit call
    occurs here. This adapter does not change the legacy centralized simulator.
    """
    root = Path(root)
    settings = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    if split not in SPLITS or not 0 <= client_id < settings["config"]["n_clients"]:
        raise ValueError("invalid client or split")
    directory = root / f"client_{client_id:03d}"
    if settings.get("schema_version", 1) >= 2:
        return load_local_split(directory, split)
    return _load_split(directory, settings, split)


def _load_split(directory, settings, split):
    if split not in SPLITS:
        raise ValueError("invalid split")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("preprocessing_sha256") and
            _file_hash(directory / "preprocessing.json") != manifest["preprocessing_sha256"]):
        raise PreprocessingError("client preprocessing configuration checksum mismatch")
    file = directory / f"{split}.jsonl"
    if _file_hash(file) != manifest["splits"][split]["sha256"]:
        raise PreprocessingError("client data checksum mismatch")
    texts, labels = [], []
    with file.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            normalized = normalize_url(row["url"])
            if not normalized.is_valid or normalized.registrable_domain != row["domain"]:
                raise PreprocessingError("client URL normalization mismatch")
            texts.append(normalized.feature_text)
            labels.append(binary_label(row["label"]))
    if len(labels) != manifest["splits"][split]["n_records"]:
        raise PreprocessingError("client record count mismatch")
    features = dict(settings["features"])
    features["ngram_range"] = tuple(features["ngram_range"])
    if features.pop("norm") != "l2" or features.pop("alternate_sign") is not False:
        raise PreprocessingError("unsupported feature transformation")
    matrix = build_vectorizer(**features).transform(texts)
    labels = np.asarray(labels, dtype=np.int8)
    return matrix, labels if split == "train" else mark_protected(labels)
