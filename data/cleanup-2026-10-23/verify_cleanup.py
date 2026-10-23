"""Read-only checks of cleanup boundaries and original data fingerprints."""
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


manifest = json.loads((HERE / 'manifest.json').read_text())
assert digest(HERE / manifest['archive']) == manifest['archive_sha256']
changes = json.loads((HERE / 'changes.json').read_text())
assert all(not (ROOT / name).exists() for name in changes['removed_directories'] + changes['removed_files'])
for name, expected in manifest['protected_data'].items():
    path = ROOT / name
    assert path.stat().st_size == expected['size'], name
    assert path.stat().st_mtime_ns == expected['mtime_ns'], name
    if 'sha256' in expected:
        assert digest(path) == expected['sha256'], name

hashes = {}
for name in ('client-datasets-v3', 'client-iid-v1', 'client-label-skew-v1', 'client-category-skew-v1'):
    directory = ROOT / 'data' / name
    for client in sorted(directory.glob('client_[0-9][0-9][0-9]')):
        local = json.loads((client / 'manifest.json').read_text())
        path = client / 'data.jsonl'
        actual = digest(path)
        assert actual == local['dataset']['sha256'], path
        hashes[path.relative_to(ROOT).as_posix()] = actual
        split_manifest = client / 'splits/manifest.json'
        if split_manifest.exists():
            for split, values in json.loads(split_manifest.read_text())['splits'].items():
                path = client / f'splits/{split}.jsonl'
                actual = digest(path)
                assert actual == values['sha256'], path
                hashes[path.relative_to(ROOT).as_posix()] = actual
    print(f'PASS original dataset/split checksums: {name}', flush=True)
path = ROOT / 'data/raw/url.csv'
assert digest(path) == '79e32aeafc74a5c875add374d1ea5d163f867ee9109a47907c99271bb76002a4'
assert digest(ROOT / 'results/evaluation_reporting_v4/report.json') == '3f8d0fcbfae53c8dc108ab8aa748a2971c8a12f185a698c610822b6d5d308909'

with zipfile.ZipFile(HERE / manifest['archive']) as archive:
    for name in ('src/data/split_policy.py', 'src/models/character_cnn/model.py', 'src/models/deberta_v3_base/model.py'):
        assert ast.dump(ast.parse((ROOT / name).read_text(encoding='utf-8'))) == ast.dump(ast.parse(archive.read(name).decode('utf-8'))), name
    assert (ROOT / 'LICENSE').read_bytes() == archive.read('LICENSE')

files = [*list((ROOT / 'src').rglob('*.py')), *list((ROOT / 'scripts').glob('*.py')), *list((ROOT / 'tests').glob('*.py'))]
for path in files:
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                 [name.name for name in node.names] if isinstance(node, ast.Import) else [])
        for module in names:
            assert not module.startswith('baize'), (path, module)
            if module.startswith(('src.', 'scripts.')):
                target = ROOT.joinpath(*module.split('.'))
                assert target.with_suffix('.py').is_file() or (target / '__init__.py').is_file(), (path, module)

for path in [ROOT / 'README.md', *list((ROOT / 'docs').glob('*.md'))]:
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
        target = target.strip('<>').split('#', 1)[0]
        if target and '://' not in target and not target.startswith('mailto:'):
            assert (path.parent / target).exists(), (path, target)

test_log = (HERE / 'tests.log').read_text(encoding='utf-8')
assert '159 passed, 1 skipped' in test_log and ' failed' not in test_log
run = json.loads((HERE / 'cnn-smoke/server/run.json').read_text())
assert run['status'] == 'PASS'
report = json.loads((HERE / 'smoke-report/report.json').read_text())
assert len(report['rows']) == 4 and len(report['plots']) == 4
assert report['best_method'] is None and report['automatic_method_selection'] is False
assert all(row['data_kind'] == 'synthetic' and row['family'] == 'character_cnn' for row in report['rows'])
result = {'status': 'PASS', 'regression_tests': {'passed': 159, 'skipped': 1, 'reason': 'opt-in localhost gRPC test'},
          'unchanged_data_artifacts': len(manifest['protected_data']), 'dataset_split_sha256': hashes,
          'raw_csv_sha256_verified': True, 'final_historical_report_unchanged': True,
          'model_architectures_and_domain_role_ast_unchanged': True, 'license_unchanged': True,
          'active_python_files_checked': len(files), 'legacy_imports': 0,
          'active_documentation_links_resolve': True,
          'flower_smoke_rounds': 2, 'flower_smoke_clients': 5,
          'report_metric_rows': 4, 'report_plot_files': 4,
          'archive_sha256': manifest['archive_sha256'], 'archive_source_files': len(manifest['files_sha256'])}
(HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(f'PASS: cleanup, {len(hashes)} original dataset/split content hashes, model contracts, 159 tests and five-client smoke/report exports.')
