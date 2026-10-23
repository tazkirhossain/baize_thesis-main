"""Exercise the safety policy in synthetic disposable Git indexes, never the corpus."""
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.check_repository_safety import check_repository


@pytest.fixture
def repository(tmp_path):
    if not shutil.which("git"):
        pytest.skip("Git is unavailable")
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / ".gitignore", tmp_path / ".gitignore")
    return tmp_path


def stage(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    subprocess.run(["git", "-C", str(root), "add", "-f", "--", name], check=True, capture_output=True)


@pytest.mark.parametrize("name", ["data/raw/url.csv", "leaked.csv", "clients/client_000/data.jsonl",
    ".env", "private.pem", "model.pt", "checkpoints/latest.bin", "other/exclusions.jsonl",
    "archive.zip", "somewhere/.aws/credentials"])
def test_force_added_sensitive_artifacts_are_blocked(repository, name):
    stage(repository, name, "synthetic placeholder")
    assert any(path == name for path, _ in check_repository(repository))


def test_source_code_and_sanitized_examples_are_allowed(repository):
    for name in ("src/data/preprocessing.py", "src/models/character_cnn/model.py", ".env.example"):
        stage(repository, name, "# no sensitive values")
    assert check_repository(repository) == []


def test_private_key_in_unexpected_extension_is_blocked(repository):
    stage(repository, "notes.txt", "-----BEGIN " + "PRIVATE KEY-----\nfake test fixture\n")
    assert any("private key" in reason for _, reason in check_repository(repository))


def test_jsonl_exports_have_no_legacy_metric_exception(repository):
    name = "experiments/results/E1_random_seeds.jsonl"
    stage(repository, name, '{"seed":0,"pr_auc":0.4,"url":"synthetic.invalid"}\n')
    assert check_repository(repository)
    stage(repository, name, '{"seed":0,"pr_auc":0.4}\n')
    assert check_repository(repository)


@pytest.mark.parametrize("prefix,suffix", [("AKIA", "A"*16), ("ghp_", "x"*36), ("sk-proj-", "x"*48)])
def test_credential_like_tokens_in_source_are_blocked_without_echoing(repository, prefix, suffix):
    value = prefix + suffix  # Synthetic patterns, not actual credentials.
    stage(repository, "unsafe.txt", value)
    results = check_repository(repository)
    assert any("token" in reason for _, reason in results)
    assert value not in str(results)
