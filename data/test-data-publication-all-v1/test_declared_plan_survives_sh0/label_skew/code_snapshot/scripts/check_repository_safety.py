"""Owner-approved data publication guard: credentials blocked, large files use LFS.

Datasets, generated artifacts and dependencies were explicitly authorized for
publication on 23 October 2026. Findings print paths/reasons, never secret values.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

PRIVATE_KEY = re.compile(rb'(?m)^-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----')
TOKENS = [re.compile(pattern) for pattern in (
    rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b', rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b')]


def synthetic_fixture(name, content):
    """Only exact, visibly fake historical safety-test payloads are exempt."""
    if not name.startswith('data/test-') or len(content)>128:
        return False
    normalized = content.replace(b'\r\n', b'\n')
    return normalized in {
        b'synthetic placeholder', b'AKIA'+b'A'*16, b'ghp_'+b'x'*36,
        b'sk-proj-'+b'x'*48, b'-----BEGIN PRIVATE KEY-----\nfake test fixture\n'}


def credential_name(name):
    path = Path(name)
    return (path.name in ('id_rsa','id_ed25519','credentials.json','secrets.json','secrets.yaml','secrets.yml','secrets.toml')
            or (path.name.startswith('.env') and path.name not in ('.env.example','.env.template'))
            or '/.aws/' in '/'+name or '/.ssh/' in '/'+name or '/.azure/' in '/'+name
            or path.suffix.lower() in ('.pem','.key','.p12','.pfx','.jks','.keystore'))


def content_findings(path, name):
    with path.open('rb') as stream:
        first = stream.read(129)
        if synthetic_fixture(name, first):
            return []
        if credential_name(name) and not (path.suffix.lower()=='.pem' and b'-----BEGIN CERTIFICATE-----' in first):
            return [(name, 'credential/private-key filename requires exclusion; value withheld')]
        stream.seek(0)
        carry = b''
        while True:
            block = stream.read(4*1024**2)
            if not block:
                return []
            value = carry+block
            if PRIVATE_KEY.search(value):
                return [(name, 'private key material detected; value withheld')]
            if any(pattern.search(value) for pattern in TOKENS):
                return [(name, 'credential-like token detected; value withheld')]
            carry = value[-512:]


def check_repository(root: Path) -> list[tuple[str, str]]:
    tracked = subprocess.run(['git','-C',str(root),'ls-files','--stage','-z'], capture_output=True,check=True).stdout
    entries = []
    for record in tracked.split(b'\0'):
        if record:
            header, name = record.split(b'\t',1)
            mode, oid, stage = header.split()
            if stage != b'0' or mode == b'160000':
                raise RuntimeError('unmerged/submodule entries cannot be published as complete data')
            entries.append((oid,name.decode('utf-8')))
    if not entries:
        return []
    ids = sorted({oid for oid,_ in entries})
    checked = subprocess.run(['git','-C',str(root),'cat-file','--batch-check=%(objectname) %(objectsize)'],
        input=b'\n'.join(ids)+b'\n',capture_output=True,check=True).stdout
    sizes = {line.split()[0]:int(line.split()[1]) for line in checked.splitlines()}
    findings = []
    for oid,name in entries:
        if sizes[oid] > 100*1024**2:
            findings.append((name,'regular Git blob exceeds 100 MiB; use Git LFS'))
        path = root/name
        if path.is_file():
            findings.extend(content_findings(path,name))
    return findings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    try:
        findings=check_repository(args.root.resolve())
    except (OSError,subprocess.CalledProcessError,RuntimeError):
        print('NOT VERIFIED: Git index or publication-policy inspection failed.')
        return 2
    for path,reason in findings:
        print(f'BLOCKED: {path}: {reason}')
    if findings:
        return 1
    print('PASS: authorized datasets/artifacts allowed; no detected credentials/private keys; file limits checked. Remote history not checked.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
