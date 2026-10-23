"""Single-URL classification for the local demo frontend.

This is a visualisation of the research pipeline, not a product. The live
checker reuses the same components the experiments use:

    raw string -> normalize_url (PRIVACY.md §3) -> HashingVectorizer
    -> LogisticSGD.predict_proba -> abstention band around 0.5

It never fetches the URL. Query, fragment, userinfo, and port are stripped
before scoring and are not returned to the browser.

Model source, in order:
1. A previously saved `data/models/classifier.npz` (gitignored).
2. Else a uniform sample of `data/raw/url.csv` if that file exists locally.
3. Else the synthetic fixture. Abstract markers only (LABELING_POLICY.md §7);
   a synthetic model will not classify real adult sites.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from baize.data.dataset import URLDataset, make_synthetic_dataset
from baize.data.features import build_vectorizer
from baize.data.url import normalize_url, scrub_url_secrets, structural_features
from baize.models.linear import LogisticSGD
from baize.provenance import git_commit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "data" / "models" / "classifier.npz"
DEFAULT_CORPUS_PATH = REPO_ROOT / "data" / "raw" / "url.csv"

MAX_URL_CHARS = 2048
ABSTAIN_LOW = 0.40
ABSTAIN_HIGH = 0.60
DEMO_N_FEATURES = 2**16
DEMO_NGRAM_RANGE = (3, 5)

_SYNTHETIC_LIMITATIONS = (
    "Trained on the synthetic fixture (abstract markers, not real adult "
    "vocabulary). It will not classify real websites. Place the corpus at "
    "data/raw/url.csv and restart to train a local corpus demo. That still is "
    "not the thesis 2^18-feature / 35-epoch model."
)
_CORPUS_LIMITATIONS = (
    "Trained on a uniform sample of the local url.csv. Labels are a pre-~2010 "
    "directory taxonomy (Adult vs 14 other categories), not a live crawl. "
    "The page is never fetched. Query strings are discarded before scoring. "
    "This is a demo, not the archived experiment checkpoint."
)


@dataclass(frozen=True)
class Classification:
    ok: bool
    verdict: str
    p_adult: float | None
    logit: float | None
    reason: str
    scheme: str
    host: str
    path: str
    registrable_domain: str
    feature_text: str
    scrubbed_url: str
    query_or_userinfo_stripped: bool
    n_features: int
    ngram_range: tuple[int, int]
    nnz: int
    structural: dict[str, float]
    abstain_band: tuple[float, float]
    threshold: float
    model_source: str
    fetched: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ngram_range"] = list(self.ngram_range)
        d["abstain_band"] = list(self.abstain_band)
        return d


class UrlClassifier:
    """Hashing n-gram logistic classifier with a JSON-serialisable prediction."""

    def __init__(
        self,
        model: LogisticSGD,
        n_features: int,
        ngram_range: tuple[int, int],
        meta: dict,
    ) -> None:
        self.model = model
        self.n_features = int(n_features)
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.meta = dict(meta)
        self.vectorizer = build_vectorizer(
            kind="hashing",
            n_features=self.n_features,
            ngram_range=self.ngram_range,
        )

    @property
    def source(self) -> str:
        return str(self.meta.get("source", "unknown"))

    def classify(self, raw: str, threshold: float = 0.5) -> Classification:
        original = "" if raw is None else str(raw)
        stripped = original != scrub_url_secrets(original)
        if len(original) > MAX_URL_CHARS:
            return self._invalid("", "too_long", stripped=True)


        n = normalize_url(original)
        if not n.is_valid:
            return self._invalid(n.raw, n.reason or "invalid", stripped=stripped)

        X = self.vectorizer.transform([n.feature_text])
        logit = float(self.model.decision_function(X)[0])
        p = float(self.model.predict_proba(X)[0])
        if p < ABSTAIN_LOW:
            verdict = "non_adult"
        elif p > ABSTAIN_HIGH:
            verdict = "adult"
        else:
            verdict = "uncertain"
        # Hard threshold still reported; abstention overrides the stamp.
        _ = threshold
        nnz = int(X.nnz)
        return Classification(
            ok=True,
            verdict=verdict,
            p_adult=p,
            logit=logit,
            reason="",
            scheme=n.scheme,
            host=n.host,
            path=n.path,
            registrable_domain=n.registrable_domain,
            feature_text=n.feature_text,
            scrubbed_url=n.raw,
            query_or_userinfo_stripped=stripped,
            n_features=self.n_features,
            ngram_range=self.ngram_range,
            nnz=nnz,
            structural=structural_features(n),
            abstain_band=(ABSTAIN_LOW, ABSTAIN_HIGH),
            threshold=float(threshold),
            model_source=self.source,
        )

    def _invalid(self, scrubbed: str, reason: str, stripped: bool) -> Classification:
        return Classification(
            ok=False,
            verdict="invalid",
            p_adult=None,
            logit=None,
            reason=reason,
            scheme="",
            host="",
            path="",
            registrable_domain="",
            feature_text="",
            scrubbed_url=scrubbed,
            query_or_userinfo_stripped=stripped,
            n_features=self.n_features,
            ngram_range=self.ngram_range,
            nnz=0,
            structural={},
            abstain_band=(ABSTAIN_LOW, ABSTAIN_HIGH),
            threshold=0.5,
            model_source=self.source,
        )

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path or DEFAULT_MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w=self.model.w,
            b=np.array([self.model.b], dtype=np.float64),
            n_features=np.array([self.n_features], dtype=np.int64),
            ngram_lo=np.array([self.ngram_range[0]], dtype=np.int64),
            ngram_hi=np.array([self.ngram_range[1]], dtype=np.int64),
        )
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "UrlClassifier":
        path = Path(path or DEFAULT_MODEL_PATH)
        with np.load(path) as blob:
            n_features = int(blob["n_features"][0])
            ngram = (int(blob["ngram_lo"][0]), int(blob["ngram_hi"][0]))
            model = LogisticSGD(n_features=n_features, l2=1e-7, seed=0)
            w = np.asarray(blob["w"], dtype=np.float64)
            b = float(np.asarray(blob["b"]).reshape(-1)[0])
        if w.shape != (n_features,):
            raise ValueError(f"weight length {w.shape} != n_features {n_features}")
        model.set_params(np.concatenate([w, np.array([b], dtype=np.float64)]))
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"source": "unknown"}
        return cls(model=model, n_features=n_features, ngram_range=ngram, meta=meta)


def train_classifier(
    ds: URLDataset,
    *,
    n_features: int = DEMO_N_FEATURES,
    ngram_range: tuple[int, int] = DEMO_NGRAM_RANGE,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 8.0,
    l2: float = 1e-7,
    seed: int = 0,
    source: str = "synthetic",
    limitations: str = _SYNTHETIC_LIMITATIONS,
    extra_meta: dict | None = None,
) -> UrlClassifier:
    if len(ds) == 0:
        raise ValueError("cannot train on an empty dataset")
    vec = build_vectorizer(kind="hashing", n_features=n_features, ngram_range=ngram_range)
    X = vec.transform(ds.texts)
    model = LogisticSGD(n_features=n_features, l2=l2, seed=seed)
    rec = model.fit(X, ds.labels, epochs=epochs, batch_size=batch_size, lr=lr)
    meta = {
        "source": source,
        "n_train": int(len(ds)),
        "positive_rate": float(ds.positive_rate),
        "class_counts": {str(k): int(v) for k, v in ds.class_counts.items()},
        "n_features": int(n_features),
        "ngram_range": list(ngram_range),
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "l2": float(l2),
        "final_loss": rec.get("final_loss"),
        "git_commit": git_commit(),
        "limitations": limitations,
        "fetches_pages": False,
        "dataset_name": ds.name,
        "dataset_fingerprint": ds.fingerprint(),
    }
    if extra_meta:
        meta.update(extra_meta)
    return UrlClassifier(model=model, n_features=n_features, ngram_range=ngram_range, meta=meta)


def bootstrap_classifier(
    *,
    model_path: Path | str | None = None,
    corpus_path: Path | str | None = None,
    force_synthetic: bool = False,
    corpus_sample: int = 80_000,
    retrain: bool = False,
) -> UrlClassifier:
    """Load a saved demo model, else train one (corpus sample or synthetic)."""
    model_path = Path(model_path or DEFAULT_MODEL_PATH)
    corpus_path = Path(corpus_path or DEFAULT_CORPUS_PATH)

    if not retrain and model_path.exists():
        return UrlClassifier.load(model_path)

    if not force_synthetic and corpus_path.exists():
        from baize.data.loaders import sample_url_csv

        ds, _rep = sample_url_csv(str(corpus_path), n=corpus_sample, seed=0, name="url-csv-demo")
        clf = train_classifier(
            ds,
            n_features=DEMO_N_FEATURES,
            epochs=8,
            batch_size=512,
            lr=40.0,
            source="corpus-sample",
            limitations=_CORPUS_LIMITATIONS,
            extra_meta={"corpus_path": str(corpus_path), "corpus_sample": int(corpus_sample)},
        )
        clf.save(model_path)
        return clf

    ds = make_synthetic_dataset(n_domains=800, seed=7, name="synthetic-demo")
    clf = train_classifier(
        ds,
        n_features=DEMO_N_FEATURES,
        epochs=14,
        batch_size=256,
        lr=8.0,
        source="synthetic",
        limitations=_SYNTHETIC_LIMITATIONS,
    )
    clf.save(model_path)
    return clf
