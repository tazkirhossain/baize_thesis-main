"""Owner-approved data publication guard: credentials blocked, large files use LFS.

Datasets, generated artifacts and dependencies were explicitly authorized for
publication on 23 October 2026. Findings print paths/reasons, never secret values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
        public_certificate = (path.suffix.lower()=='.pem' and
                              b'-----BEGIN CERTIFICATE-----' in first+stream.read(64*1024))
        if credential_name(name) and not public_certificate:
            return [(name, 'credential/private-key filename requires exclusion; value withheld')]
        stream.seek(0)
        carry = b''
        while True:
            block = stream.read(4*1024**2)
            if not block:
                return []
            value = carry+block
            if name.startswith('data/test-') and name.endswith('.log'):
                # Pytest parameter IDs can echo these exact non-issued mock
                # values. Remove only the known mock spellings, never arbitrary
                # token-shaped strings or an entire log from inspection.
                value = value.replace(b'AKIA'+b'A'*16,b'').replace(b'ghp_'+b'x'*36,b'')
                value = value.replace(b'sk-proj--'+b'x'*48,b'').replace(b'sk-proj-'+b'x'*48,b'')
            if b'-----BEGIN ' in value and PRIVATE_KEY.search(value):
                return [(name, 'private key material detected; value withheld')]
            matches_token = (
                ((b'AKIA' in value or b'ASIA' in value) and TOKENS[0].search(value))
                or (any(prefix in value for prefix in (b'ghp_',b'gho_',b'ghu_',b'ghs_',b'ghr_')) and TOKENS[1].search(value))
                or (b'sk-' in value and TOKENS[2].search(value)))
            if matches_token:
                return [(name, 'credential-like token detected; value withheld')]
            carry = value[-512:]


def check_repository(root: Path, *, progress=False) -> list[tuple[str, str]]:
    reviewed = {}
    review_path = root/'data/publication-nonsecret-review.json'
    if review_path.exists():
        record = json.loads(review_path.read_text(encoding='utf-8'))
        if record.get('actual_credentials_allowed') is not False:
            raise RuntimeError('publication review cannot authorize actual credentials')
        for item in record['reviewed_files']:
            name, digest = item['path'], item['sha256']
            if (Path(name).is_absolute() or '..' in Path(name).parts
                    or not re.fullmatch('[0-9a-f]{64}',digest)
                    or item['kind'] not in ('published_vendor_test_fixture','exact_known_synthetic_safety_artifact')):
                raise RuntimeError('invalid exact-file non-secret review')
            reviewed[name] = digest
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
    inspected_content = {}
    if progress:
        print(f'Inspecting {len(entries):,} tracked files.',flush=True)
    for index,(oid,name) in enumerate(entries,1):
        if sizes[oid] > 100*1024**2:
            findings.append((name,'regular Git blob exceeds 100 MiB; use Git LFS'))
        path = root/name
        if path.is_file():
            if name in reviewed:
                if hashlib.sha256(path.read_bytes()).hexdigest()!=reviewed[name]:
                    findings.append((name,'reviewed public/synthetic fixture changed; review required'))
                continue
            key = (oid, name.startswith('data/test-'))
            # Historical source snapshots contain tens of thousands of exact
            # duplicate Git objects. Reuse content findings; path-based checks
            # and the narrowly scoped synthetic-fixture exemption stay separate.
            if credential_name(name) or key not in inspected_content:
                result = content_findings(path,name)
                if not credential_name(name):
                    inspected_content[key] = [reason for _,reason in result]
            else:
                result = [(name,reason) for reason in inspected_content[key]]
            findings.extend(result)
        if progress and index%5000==0:
            print(f'Inspected {index:,}/{len(entries):,} files.',flush=True)
    return findings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    try:
        findings=check_repository(args.root.resolve(),progress=True)
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
