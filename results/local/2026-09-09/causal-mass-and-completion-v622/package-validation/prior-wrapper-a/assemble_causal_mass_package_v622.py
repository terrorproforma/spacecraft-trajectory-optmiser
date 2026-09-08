"""Assemble a fresh, unsealed v622 stage from a pinned inventory. CPU only."""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import tarfile

from prepare_causal_mass_package_v622 import ROOT, relative, safe_bytes


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_tar(path, entries):
    members = {}
    with path.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as archive:
            for name, data in sorted(entries.items()):
                safe_bytes(name, data)
                member = tarfile.TarInfo(relative(name))
                member.size = len(data)
                member.mode = 0o644
                member.mtime = 0
                archive.addfile(member, io.BytesIO(data))
                members[name] = {'bytes': len(data), 'sha256': sha(data)}
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source_inventory = args.inventory.read_bytes()
    inventory = json.loads(source_inventory)
    assert inventory['state'] == 'draft_inventory_only' and inventory['sealed'] is False
    output = args.out.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'sources/archives').mkdir(parents=True)
    (output / 'raw').mkdir()
    (output / 'reproduce').mkdir()
    file_map, objects, raw_entries = {}, {}, {}

    def get_source(origin, expected):
        path = ROOT / relative(origin)
        assert path.is_file() and not path.is_symlink()
        data = path.read_bytes()
        safe_bytes(origin, data)
        assert len(data) == expected['bytes'] and sha(data) == expected['sha256'], origin
        return data

    # Every original archive remains byte-exact and is stored once by archive SHA.
    for digest, entry in inventory['source_archives'].items():
        data = get_source(entry['original_paths'][0], {'bytes': entry['bytes'], 'sha256': digest})
        stored = 'sources/archives/' + digest + '.tar.gz'
        (output / stored).write_bytes(data)
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as archive:
            actual = {}
            for member in archive:
                assert member.isfile() or member.isdir()
                name = relative(member.name)
                if member.isdir():
                    continue
                payload = archive.extractfile(member).read()
                safe_bytes(name, payload)
                value = {'bytes': len(payload), 'sha256': sha(payload)}
                assert value == entry['members'][name], name
                actual[name] = value
                objects.setdefault(value['sha256'], payload)
            assert actual == entry['members']
        entry['stored_path'] = stored

    for group, members in inventory['source_snapshots'].items():
        for name, entry in members.items():
            payload = get_source(group + '/' + name, entry)
            objects.setdefault(entry['sha256'], payload)
    assert set(objects) == set(inventory['source_objects'])
    for digest, payload in objects.items():
        assert sha(payload) == digest and len(payload) == inventory['source_objects'][digest]['bytes']
    object_members = write_tar(output / 'sources/objects.tar.gz', objects)

    for destination, expected in inventory['artifacts'].items():
        data = get_source(expected['origin'], expected)
        record = dict(expected)
        digest = expected['sha256']
        if digest in inventory['source_archives']:
            record['stored_path'] = inventory['source_archives'][digest]['stored_path']
        elif ((data and len(data) > 131072 and Path(destination).name != 'report.json'
               and any(part in ('run', 'readbacks', 'output') for part in Path(destination).parts))
              or Path(destination).suffix in ('.npz', '.npy')):
            raw_entries[destination] = data
            record.update(stored_archive='raw/readbacks.tar.gz', archive_member=destination)
        else:
            path = output / relative(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            record['stored_path'] = destination
        file_map[destination] = record
    raw_members = write_tar(output / 'raw/readbacks.tar.gz', raw_entries)
    records = {
        'schema': 'v622-portable-evidence-v1', 'sealed': False,
        'source_inventory_sha256': sha(source_inventory), 'files': file_map,
        'source_snapshots': inventory['source_snapshots'],
        'source_archives': inventory['source_archives'], 'source_objects': object_members,
        'source_objects_archive': {'path': 'sources/objects.tar.gz', 'sha256': sha((output / 'sources/objects.tar.gz').read_bytes())},
        'raw_archive': {'path': 'raw/readbacks.tar.gz', 'sha256': sha((output / 'raw/readbacks.tar.gz').read_bytes()), 'members': raw_members},
        'components': inventory['components'],
        'mass_source_archive_sha256': '1b64328171dfe6cb1623fb75739006abfa2f31433c5ad2b12f92d766bd949511',
    }
    (output / 'portable-manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    (output / 'draft-inventory.json').write_bytes(source_inventory)
    (output / '.gitattributes').write_text('* -text\n')
    (output / 'STATUS.json').write_text(json.dumps({
        'sealed': False, 'state': 'draft_evidence_stage', 'GPU_calls': 0, 'Git_calls': 0,
        'pending': ['Root final benchmark/review pins', 'Portable sequential CPU checks', 'Final README and index'],
        'historical_paths': 'Original report/log bytes retain historical paths; all new materialization paths are repository-relative.',
    }, indent=2) + '\n')
    for name in ('prepare_causal_mass_package_v622.py', 'assemble_causal_mass_package_v622.py',
                 'portable_causal_mass_v622.py', 'recheck_causal_mass_v622.py'):
        path = Path(__file__).parent / name
        if path.is_file():
            (output / 'reproduce' / name).write_bytes(path.read_bytes())
    print(json.dumps({'sealed': False, 'files': len(file_map), 'source_objects': len(objects),
                      'source_archives': len(records['source_archives']), 'raw_members': len(raw_members),
                      'output': output.relative_to(ROOT).as_posix()}))


if __name__ == '__main__':
    main()
