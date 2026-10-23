"""Download and verify the four public, pinned DeBERTa-v3-base PyTorch files.

No client data or credentials are sent. Model assets stay under ignored data/.
This command prepares pretrained assets; it does not train a URL classifier.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("config.json", "tokenizer_config.json", "spm.model", "pytorch_model.bin")


def file_hash(path, algorithm="sha256", *, git_blob=False):
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / "data"):
        raise ValueError("pretrained files must remain under ignored repository data/")
    cfg = json.loads((ROOT / "config/models/deberta_v3_base.json").read_text())
    revision = cfg["revision"]
    if cfg["source"] != "microsoft/deberta-v3-base" or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("expected the public Microsoft model and an immutable revision")
    # Use ordinary resumable HTTPS and local cache; never use an implicit token.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi(token=False).model_info(cfg["source"], revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("Hub revision does not match the pinned checkpoint")
    siblings = {item.rfilename: item for item in info.siblings}
    if set(FILES) - siblings.keys():
        raise ValueError("the pinned checkpoint is missing required files")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "PASS", "repo_id": cfg["source"], "revision": revision,
              "public_download_token_used": False, "files": {}}
    for name in FILES:
        remote = siblings[name]
        print(f"Preparing {name}: {remote.size:,} bytes", flush=True)
        path = Path(hf_hub_download(repo_id=cfg["source"], filename=name,
                    revision=revision, local_dir=output, token=False))
        actual = file_hash(path)
        lfs_sha256 = getattr(remote.lfs, "sha256", None)
        if path.stat().st_size != remote.size:
            raise ValueError(f"downloaded size mismatch: {name}")
        if lfs_sha256:
            if actual != lfs_sha256:
                raise ValueError(f"downloaded SHA-256 mismatch: {name}")
        elif file_hash(path, "sha1", git_blob=True) != remote.blob_id:
            raise ValueError(f"downloaded Git blob mismatch: {name}")
        report["files"][name] = {"size": remote.size, "sha256": actual,
                "hub_blob_id": remote.blob_id, "hub_lfs_sha256": lfs_sha256}
        print(f"Verified {name}: SHA-256 {actual}", flush=True)
    temporary = output / "download_manifest.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "download_manifest.json")
    print(f"PASS: all four pinned files verified in {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/pretrained/deberta-v3-base")
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
