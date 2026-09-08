"""Verify/materialize v622 saved evidence without Git, native imports or GPU work."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import tarfile

from prepare_causal_mass_package_v622 import relative, safe_bytes


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_bytes(path, expected):
    result = {}
    with tarfile.open(path, 'r:*') as archive:
        for member in archive:
            assert member.isfile(), member.name
            name = relative(member.name)
            assert name not in result
            value = archive.extractfile(member).read()
            safe_bytes(name, value)
            assert {'bytes': len(value), 'sha256': digest(value)} == expected[name], name
            result[name] = value
    assert set(result) == set(expected)
    return result


def verify(package):
    package = Path(package).resolve()
    manifest = json.loads((package / 'portable-manifest.json').read_text())
    assert manifest['schema'] == 'v622-portable-evidence-v1'
    object_archive = package / relative(manifest['source_objects_archive']['path'])
    assert digest(object_archive.read_bytes()) == manifest['source_objects_archive']['sha256']
    objects = archive_bytes(object_archive, manifest['source_objects'])
    assert all(digest(data) == name for name, data in objects.items())
    raw = manifest['raw_archive']
    raw_path = package / relative(raw['path'])
    assert digest(raw_path.read_bytes()) == raw['sha256']
    readbacks = archive_bytes(raw_path, raw['members'])
    for digest_key, entry in manifest['source_archives'].items():
        path = package / relative(entry['stored_path'])
        content = path.read_bytes()
        assert len(content) == entry['bytes'] and digest(content) == digest_key
        actual = {}
        with tarfile.open(fileobj=io.BytesIO(content), mode='r:*') as archive:
            for member in archive:
                assert member.isfile() or member.isdir()
                name = relative(member.name)
                if member.isdir():
                    continue
                assert name not in actual
                value = archive.extractfile(member).read()
                row = {'bytes': len(value), 'sha256': digest(value)}
                assert row == entry['members'][name]
                assert value == objects[row['sha256']]
                actual[name] = row
        assert actual == entry['members']
    for name, row in manifest['files'].items():
        value = (package / relative(row['stored_path'])).read_bytes() if 'stored_path' in row else readbacks[row['archive_member']]
        safe_bytes(name, value)
        assert len(value) == row['bytes'] and digest(value) == row['sha256'], name
    mass = json.loads((package / manifest['files']['mass/build/a/manifest.json']['stored_path']).read_text())
    assert mass['source_archive_sha256'] == manifest['mass_source_archive_sha256']
    members = manifest['source_archives'][mass['source_archive_sha256']]['members']
    assert {name: row['sha256'] for name, row in members.items()} == mass['source_sha256']
    base = mass['base_source_sha256']
    assert all(value in objects for value in base.values())
    assert all(mass['source_sha256'][name] == value for name, value in base.items() if name not in mass['owned_paths'])
    assert set(mass['source_sha256']) - set(base) == set(mass['owned_paths']) - set(base)
    assert digest(''.join(name + ':' + value + '\n' for name, value in mass['source_sha256'].items()).encode()) == mass['source_tree_sha256']
    return manifest, objects, readbacks, {
        'passed': True, 'files': len(manifest['files']), 'source_objects': len(objects),
        'source_archives': len(manifest['source_archives']), 'raw_members': len(readbacks),
        'mass_source_tree_sha256': mass['source_tree_sha256'],
        'original_baseline_file_bytes_recoverable': len(base),
        'source_proof_scope': 'Offline exact file/map/overlay proof; original recorded Git-base verification is retained, not rerun.',
        'GPU_calls': 0, 'native_library_loads': 0, 'Git_calls': 0,
    }


def materialize(package, out):
    manifest, objects, readbacks, report = verify(package)
    package, out = Path(package).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)

    def write(name, data):
        path = out / relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == data, name
        else:
            path.write_bytes(data)

    for row in manifest['files'].values():
        value = (package / row['stored_path']).read_bytes() if 'stored_path' in row else readbacks[row['archive_member']]
        write(row['origin'], value)
    for root, rows in manifest['source_snapshots'].items():
        for name, row in rows.items():
            write(root + '/' + name, objects[row['sha256']])
    mass_members = manifest['source_archives'][manifest['mass_source_archive_sha256']]['members']
    for name, row in mass_members.items():
        write(name, objects[row['sha256']])
    (out / 'materialization.json').write_text(json.dumps(report, indent=2) + '\n')
    return out, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = materialize(args.package, args.out)[1] if args.out else verify(args.package)[3]
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
