"""Read-only guard for tracked data/credentials; prints paths, never secret values."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def check_repository(root: Path) -> list[tuple[str, str]]:
    tracked = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                             capture_output=True, check=True).stdout
    if not tracked:
        return []
    ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", "--stdin", "-z"],
                             input=tracked, capture_output=True)
    if ignored.returncode not in (0, 1):
        raise RuntimeError("Git ignore-policy check failed")
    findings = [(name.decode("utf-8", errors="replace"), "tracked file violates .gitignore policy")
                for name in ignored.stdout.split(b"\0") if name]
    blocked = {path for path, _ in findings}
    private_key = re.compile(rb"(?m)^-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----")
    token_patterns = [re.compile(pattern) for pattern in (
        rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
        rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b")]
    metric_fields = {"seed", "best_epoch", "val_pr_auc", "pr_auc", "roc_auc", "accuracy",
                     "precision", "recall", "f1", "fpr", "fnr", "tp", "fp", "tn", "fn",
                     "n", "positive_rate", "threshold", "brier", "ece"}
    for raw_name in tracked.split(b"\0"):
        if not raw_name:
            continue
        name = raw_name.decode("utf-8", errors="replace")
        path = root / name
        if name in blocked or not path.is_file():
            continue
        if path.stat().st_size > 1_000_000:
            findings.append((name, "large tracked artifact requires explicit review"))
            continue
        content = path.read_bytes()
        if private_key.search(content):
            findings.append((name, "private key material detected"))
        if any(pattern.search(content) for pattern in token_patterns):
            findings.append((name, "credential-like token detected; value withheld"))
        # The only JSONL exceptions are pre-existing aggregate experiment metrics.
        if path.suffix.lower() == ".jsonl":
            try:
                rows = [json.loads(line) for line in content.splitlines() if line.strip()]
                if not rows or any(not isinstance(row, dict) or "pr_auc" not in row or
                                   "seed" not in row or not set(row).issubset(metric_fields) or
                                   any(type(value) not in (int, float) for value in row.values())
                                   for row in rows):
                    findings.append((name, "JSONL exception is not an aggregate metric file"))
            except (ValueError, UnicodeDecodeError):
                findings.append((name, "invalid aggregate metric JSONL"))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = check_repository(args.root.resolve())
    except (OSError, subprocess.CalledProcessError, RuntimeError):
        print("NOT VERIFIED: no readable Git index, or Git policy inspection failed.")
        return 2
    for path, reason in result:
        print(f"BLOCKED: {path}: {reason}")
    if result:
        return 1
    print("PASS: no prohibited tracked files found. Remote history is not checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
