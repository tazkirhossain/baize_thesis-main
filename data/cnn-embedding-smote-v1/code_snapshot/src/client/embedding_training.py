"""Client-local encoder warmup followed by frozen-feature head balancing."""
import tempfile
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.balancing import print_balance_report, sample_training_indices
from src.data.embedding_smote import EmbeddingTrainingSet, smote_embeddings


def train_batches(model, loader, *, epochs, learning_rate, device, seed, embeddings=False):
    model.eval() if embeddings else model.train()
    head = model.feature_head() if embeddings else model
    head.train()
    optimizer = torch.optim.SGD(head.parameters(), lr=learning_rate)
    loss_sum, seen = 0.0, 0
    with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
        torch.manual_seed(seed)
        for _ in range(epochs):
            for features, labels in loader:
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                if embeddings:
                    logits = model.classify_features(features.to(device))
                else:
                    output = model(**{key: value.to(device) for key, value in features.items()})
                    logits = output.logits if hasattr(output, "logits") else output
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite local training loss")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())*len(labels)
                seen += len(labels)
    model.zero_grad(set_to_none=True)
    return loss_sum, seen


def train_embedding_member(model, dataset, data, cfg, balance, *, collate, device, seed, selection, cache_root):
    if not all(hasattr(model, name) for name in ("extract_features", "classify_features", "feature_head")):
        raise TypeError("model does not expose a continuous encoder/head interface")
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    model.to(device)
    def loader(view, *, shuffle, encoder=True):
        return DataLoader(view, batch_size=cfg["batch_size"], shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed), num_workers=0,
            collate_fn=collate if encoder else None)
    warm_loss, warm_seen = train_batches(model, loader(dataset, shuffle=True),
        epochs=cfg["representation"]["warmup_epochs"], learning_rate=cfg["learning_rate"], device=device, seed=seed)
    model.eval()
    with ExitStack() as stack:
        temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix="embedding-train-", dir=cache_root))
        matrix, offset = None, 0
        with torch.no_grad():
            for features, labels in loader(dataset, shuffle=False):
                values = model.extract_features(**{key: value.to(device) for key, value in features.items()}).detach().cpu().numpy()
                if values.ndim != 2 or not np.isfinite(values).all():
                    raise ValueError("invalid continuous encoder embeddings")
                if matrix is None:
                    matrix = np.lib.format.open_memmap(Path(temporary)/"training.npy", mode="w+", dtype=np.float32,
                                                       shape=(len(dataset), values.shape[1]))
                    stack.callback(matrix._mmap.close)
                matrix[offset:offset+len(labels)] = values
                offset += len(labels)
        if offset != len(dataset):
            raise ValueError("training embedding cache coverage mismatch")
        matrix.flush()
        embedded = EmbeddingTrainingSet(matrix, data.labels, provenance=data.sha256)
        if balance.method == "smote":
            view, report = smote_embeddings(embedded, balance)
        else:
            selection = selection or sample_training_indices(data.labels, balance, split="train")
            view, report = selection.apply(embedded), selection.report
        print_balance_report(report)
        print(f"  encoder_embeddings; warmup={cfg['representation']['warmup_epochs']} epochs; "
              f"features={matrix.shape[1]}; SMOTE k={report.get('effective_k_neighbors', 0)}; status={report['status']}", flush=True)
        head_loss, head_seen = train_batches(model, loader(view, shuffle=True, encoder=False),
            epochs=cfg["local_epochs"], learning_rate=cfg["learning_rate"], device=device, seed=seed, embeddings=True)
        # Explicitly release the mmap before TemporaryDirectory cleanup on Windows.
        del view, embedded
        del matrix
    model.cpu()
    return warm_loss+head_loss, warm_seen+head_seen, report, warm_seen
