"""Exercise the safety policy in synthetic disposable Git indexes, never the corpus."""
import shutil
import hashlib
import json
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


@pytest.mark.parametrize("name", [".env", "private.pem", "somewhere/.aws/credentials"])
def test_force_added_credentials_are_blocked(repository, name):
    stage(repository, name, "synthetic placeholder")
    assert any(path == name for path, _ in check_repository(repository))


@pytest.mark.parametrize('name',['data/raw/url.csv','leaked.csv','clients/client_000/data.jsonl',
    'model.pt','checkpoints/latest.bin','other/exclusions.jsonl','archive.zip'])
def test_owner_authorized_data_and_artifacts_are_allowed(repository,name):
    stage(repository,name,'synthetic placeholder')
    assert check_repository(repository)==[]


def test_source_code_and_sanitized_examples_are_allowed(repository):
    for name in ("src/data/preprocessing.py", "src/models/character_cnn/model.py", ".env.example"):
        stage(repository, name, "# no sensitive values")
    assert check_repository(repository) == []


def test_private_key_in_unexpected_extension_is_blocked(repository):
    stage(repository, "notes.txt", "-----BEGIN " + "PRIVATE KEY-----\nfake test fixture\n")
    assert any("private key" in reason for _, reason in check_repository(repository))


def test_owner_authorized_jsonl_exports_are_allowed(repository):
    name = "experiments/results/E1_random_seeds.jsonl"
    stage(repository, name, '{"seed":0,"pr_auc":0.4,"url":"synthetic.invalid"}\n')
    assert check_repository(repository)==[]
    stage(repository, name, '{"seed":0,"pr_auc":0.4}\n')
    assert check_repository(repository)==[]


def test_exact_historical_synthetic_credentials_are_distinct_from_real_secrets(repository):
    name='data/test-fixture/test_credential_like_tokens_in0/unsafe.txt'
    value='AKIA'+'A'*16
    stage(repository,name,value)
    assert check_repository(repository)==[]
    stage(repository,name,value+' additional content')
    assert check_repository(repository)


def test_public_ca_certificate_is_not_a_private_key(repository):
    stage(repository,'data/runtime/cacert.pem','# public CA bundle\n'*100+'-----BEGIN CERTIFICATE-----\npublic fixture\n')
    assert check_repository(repository)==[]


def test_duplicate_secret_findings_keep_paths_and_fixture_context(repository):
    value='AKIA'+'A'*16
    for name in ('unsafe1.txt','unsafe2.txt','data/test-example/fixture.txt'):
        stage(repository,name,value)
    findings=check_repository(repository)
    assert {name for name,_ in findings}=={'unsafe1.txt','unsafe2.txt'}


def test_pytest_logs_allow_only_the_known_mock_values(repository):
    path='data/test-publication-example.log'
    stage(repository,path,'case sk-proj--'+'x'*48+' completed')
    assert check_repository(repository)==[]
    stage(repository,path,'case sk-proj--'+'x'*48+' unexpected ghp_'+'z'*36)
    assert check_repository(repository)


def test_public_artifact_review_is_scoped_to_exact_file_bytes(repository):
    value='ghp_'+'x'*36
    stage(repository,'public-test-fixture.txt',value)
    record={'actual_credentials_allowed':False,'reviewed_files':[{'path':'public-test-fixture.txt',
        'sha256':hashlib.sha256(value.encode()).hexdigest(),'kind':'exact_known_synthetic_safety_artifact'}]}
    stage(repository,'data/publication-nonsecret-review.json',json.dumps(record))
    assert check_repository(repository)==[]
    stage(repository,'public-test-fixture.txt',value+' changed')
    assert check_repository(repository)


@pytest.mark.parametrize("prefix,suffix", [("AKIA", "A"*16), ("ghp_", "x"*36), ("sk-proj-", "x"*48)])
def test_credential_like_tokens_in_source_are_blocked_without_echoing(repository, prefix, suffix):
    value = prefix + suffix  # Synthetic patterns, not actual credentials.
    stage(repository, "unsafe.txt", value)
    results = check_repository(repository)
    assert any("token" in reason for _, reason in results)
    assert value not in str(results)
