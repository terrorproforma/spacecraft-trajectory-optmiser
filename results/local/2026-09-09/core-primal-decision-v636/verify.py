"""Portable preservation/provenance check; no archived code is executed."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit(folder, expected=None):
    if not __debug__:
        raise RuntimeError('Assertions required')
    raw = (folder / 'index.json').read_bytes()
    if expected:
        assert sha(raw) == expected
    index = json.loads(raw)
    for name, item in index['top_files'].items():
        assert '/' not in name and '\\' not in name and name not in ('.', '..')
        content = (folder / name).read_bytes()
        assert len(content) == item['bytes'] and sha(content) == item['sha256']
    assert sha((folder / 'evidence.zip').read_bytes()) == index['archive_sha256']
    saved = {}
    with zipfile.ZipFile(folder / 'evidence.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == index['file_count']
        assert set(archive.namelist()) == set(index['files'])
        for member in archive.infolist():
            name = member.filename
            path = PurePosixPath(name)
            assert not path.is_absolute() and all(p not in ('', '.', '..') for p in path.parts)
            assert '\\' not in name and ':' not in name and not member.is_dir()
            assert not stat.S_ISLNK(member.external_attr >> 16)
            content = archive.read(member)
            item = index['files'][name]
            assert len(content) == item['bytes'] and sha(content) == item['sha256']
            assert not content.startswith((b'\x7fELF', b'MZ'))
            assert b'-----BEGIN PRIVATE KEY-----' not in content
            assert b'-----BEGIN OPENSSH PRIVATE KEY-----' not in content
            saved[name] = content
    assert sum(map(len, saved.values())) == index['expanded_bytes']
    prefix = 'build/performance/core-dual-correction-v636/'
    assert sha(saved[prefix + 'index.json']) == index['source_index_sha256']
    kit = json.loads(saved[prefix + 'index.json'])
    for name, item in kit['files'].items():
        content = saved[prefix + name]
        assert len(content) == item['bytes'] and sha(content) == item['sha256']
    findings = json.loads(saved[prefix + 'saved-support-a.json'])
    assert findings['complete']
    for name, digest in findings['input_sha256'].items():
        assert sha(saved[name]) == digest
    for name, item in json.loads(saved[prefix + 'reviewed-source-pins.json'])['files'].items():
        assert len(saved[name]) == item['bytes'] and sha(saved[name]) == item['sha256']
    assert len(findings['rows']) == 2
    assert all(findings['work'][k] == 0 for k in ('factorizations', 'corrections', 'solver_calls', 'GPU_calls'))
    assert all(row['blocked_count'] == 0 and not row['correction_feasibility_established']
               for row in findings['rows'])
    return {'passed': True, 'index_sha256': sha(raw), 'files': len(saved),
            'expanded_bytes': index['expanded_bytes'], 'saved_points': 2,
            'new_solver_or_GPU_calls': 0, 'archived_code_executed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--index-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(audit(args.package, args.index_sha256), indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')
