"""Portable standard-library saved hash/count/status check; no archived code runs."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit(package, expected_index=None):
    if not __debug__:
        raise RuntimeError('Python optimization disables required checks')
    raw_index = (package / 'index.json').read_bytes()
    if expected_index is not None:
        assert sha(raw_index) == expected_index
    index = json.loads(raw_index)
    for name, item in index['top_files'].items():
        assert '/' not in name and '\\' not in name and name not in ('.', '..')
        data = (package / name).read_bytes()
        assert len(data) == item['bytes'] and sha(data) == item['sha256']
    archive = (package / 'evidence.zip').read_bytes()
    assert sha(archive) == index['archive_sha256']
    saved = {}
    with zipfile.ZipFile(package / 'evidence.zip') as z:
        names = z.namelist()
        assert len(names) == len(set(names)) == index['file_count']
        assert set(names) == set(index['files'])
        for member in z.infolist():
            name = member.filename
            path = PurePosixPath(name)
            assert not path.is_absolute() and all(p not in ('', '.', '..') for p in path.parts)
            assert '\\' not in name and ':' not in name and not member.is_dir()
            mode = member.external_attr >> 16
            assert not stat.S_ISLNK(mode)
            data = z.read(member)
            assert not data.startswith((b'\x7fELF', b'MZ'))
            assert b'-----BEGIN PRIVATE KEY-----' not in data
            assert b'-----BEGIN OPENSSH PRIVATE KEY-----' not in data
            item = index['files'][name]
            assert len(data) == item['bytes'] and sha(data) == item['sha256']
            saved[name] = data
    assert sum(len(x) for x in saved.values()) == index['expanded_bytes']
    kit = json.loads(saved['index.json'])
    assert len(kit['files']) == kit['file_count']
    for name, item in kit['files'].items():
        assert len(saved[name]) == item['bytes'] and sha(saved[name]) == item['sha256']
    assert saved['REPORT.md'] == (package / 'README.md').read_bytes()
    rows = json.loads(saved['all-candidate-metrics.json'])
    local = [r for r in rows if r['hardware'] == 'local']
    other = [r for r in rows if r['hardware'] == 'h100']
    assert len(rows) == 3920 and len(local) == len(other) == 1960
    assert {r['request_sha256'] for r in local} == {r['request_sha256'] for r in other}
    assert len({r['request_sha256'] for r in local}) == 1960
    positives = [r for r in local if r['positive_weighted']]
    assert len(positives) == 1 and positives[0]['same_failed_return_request']
    assert all(r['weighted_delta_kg'] < 0 for r in local if r['new_order_vs_narrow'])
    assert not any(r['conflicts_with_new_ship24'] for r in rows)
    findings = json.loads(saved['findings.json'])
    assert findings['passed'] and all(v == 0 for v in findings['calls'].values())
    assert findings['current_Result_sha256'] == '1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da'
    for name, original in [('inputs/wide-run.py', 'wide/run.py'), ('inputs/wide-fit.json', 'wide/fit.json')]:
        assert sha(saved[name]) == findings['archive_members']['local/' + original]['sha256']
    return {'passed': True, 'index_sha256': sha(raw_index), 'archive_sha256': sha(archive),
            'files': len(saved), 'expanded_bytes': index['expanded_bytes'],
            'distinct_saved_requests': 1960, 'new_positive_orders': 0,
            'only_positive_is_existing_failed_request': True,
            'GPU_or_solver_or_propagation_calls': 0, 'archived_code_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--index-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.package, args.index_sha256)
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
