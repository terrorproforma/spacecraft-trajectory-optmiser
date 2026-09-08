"""Seal completed local evidence after its saved-data portability pass; no Git/GPU."""
from pathlib import Path
import hashlib
import io
import json
import shutil
import tarfile
import zipfile

from prepare_causal_mass_package_v622 import ROOT, safe_bytes, relative
from portable_causal_mass_v622 import verify

PACKAGE = ROOT / 'results/local/2026-09-09/causal-mass-and-completion-v622'
PERF = ROOT / 'build/performance'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def inspect(name, data):
    safe_bytes(name, data)
    if name.endswith(('.tar.gz', '.tar')):
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as archive:
            seen = set()
            for item in archive:
                assert item.isfile() or item.isdir(), (name, item.name)
                member = relative(item.name)
                if item.isdir():
                    continue
                assert member not in seen
                seen.add(member)
                inspect(member, archive.extractfile(item).read())
    elif name.endswith(('.npz', '.zip')):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist()))
            for member in archive.namelist():
                assert not member.endswith('/')
                safe_bytes(relative(member), archive.read(member))


def main():
    assert not (PACKAGE / 'sha256.json').exists(), 'Already sealed'
    saved = PERF / 'v622-portable-check-final/portable-cpu-recheck'
    checked = json.loads((saved / 'report.json').read_text())
    assert checked['passed'] and len(checked['commands']) == 8
    assert all(c['returncode'] == 0 for c in checked['commands'])
    assert checked['GPU_calls'] == checked['native_library_loads'] == checked['Git_calls'] == 0
    final_logs = PACKAGE / 'package-validation/final'
    final_logs.mkdir()
    for path in saved.iterdir():
        if path.is_file():
            shutil.copyfile(path, final_logs / path.name)
    # Preserve the earlier experiment document bytes; append root's final,
    # explicitly historical-baseline roadmap instead of silently rewriting them.
    document = ROOT / 'docs/SOTA_EXECUTION_PLAN_2026-09-09.md'
    publication = PACKAGE / 'docs/publication'
    publication.mkdir()
    shutil.copyfile(document, publication / document.name)
    write_json(publication / 'snapshot.json', {
        'source': document.relative_to(ROOT).as_posix(),
        'sha256': sha(document.read_bytes()),
        'scope': 'Final documentation only: labels the v622/v733 mission baseline as frozen; numerical/source evidence unchanged.',
    })
    mass = json.loads((PACKAGE / 'mass/build/a/manifest.json').read_text())
    combined = json.loads((PACKAGE / 'combined/build/a/manifest.json').read_text())
    compact = json.loads((PACKAGE / 'completion/build/g/report.json').read_text())
    owned = dict(combined['owned_sha256'])
    for name, value in compact['owned_sources'].items():
        assert name not in owned or owned[name] == value
        owned[name] = value
    assert len(owned) == 23
    for name, expected in owned.items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    host_map_path = PACKAGE / 'combined/smoke/host-source-sha256.json'
    host_map = json.loads(host_map_path.read_text())
    assert len(host_map) == 341
    manifest = json.loads((PACKAGE / 'portable-manifest.json').read_text())
    assert host_map == json.loads((PACKAGE / 'completion/benchmark/source-sha256.json').read_text())
    write_json(PACKAGE / 'source-scopes.json', {
        'identity_kind': 'Frozen SHA256 source trees, not synthetic Git commits',
        'mass': {'base_commit': mass['base_commit'], 'source_tree_sha256': mass['source_tree_sha256'],
                 'source_archive_sha256': mass['source_archive_sha256'], 'owned_paths': mass['owned_paths'],
                 'library_sha256': mass['library']['sha256']},
        'combined': {'base_commit': combined['base_commit'], 'source_tree_sha256': combined['source_tree_sha256'],
                     'source_archive_sha256': combined['source_archive_sha256'], 'owned_paths': combined['owned_paths'],
                     'library_sha256': combined['library']['sha256']},
        'compact_g_owned_sources': compact['owned_sources'],
        'final_owned_source_sha256_verified_against_live': owned,
        'host_runtime': {'scope': 'Separate 341-file Python/test source tree used by benchmark and combined compact smoke',
                         'source_map_sha256': sha(host_map_path.read_bytes()),
                         'source_archive_sha256': '7281c43ff8a4d30237f84d31ae7435ee36ff90eed711f1e718f5af0732ea532e'},
        'queue_revision_maps': [name for name in manifest['source_snapshots'] if 'refinement-admission-v622' in name],
        'binary_payloads_included': False,
    })
    (PACKAGE / 'reproduce' / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    verify(PACKAGE)
    manifest['sealed'] = True
    write_json(PACKAGE / 'portable-manifest.json', manifest)
    write_json(PACKAGE / 'STATUS.json', {
        'sealed': True, 'state': 'complete_local_evidence',
        'all_experiment_results_terminal': True, 'portable_saved_data_audits_passed': 8,
        'new_GPU_calls_during_packaging': 0, 'Git_calls_during_packaging': 0,
        'pending_experiments': [],
        'prior_failure_records': 'Preserved historical attempts in package-validation and completion; none is a current packaging blocker.',
        'source_scope': '23 unique owned native/Python/test files checked against their frozen source maps; unrelated concurrent files excluded.',
        'historical_paths': 'Original logs retain exact historical paths; portable materialization uses repository-relative paths.',
    })
    records = {}
    for path in sorted(PACKAGE.rglob('*')):
        assert not path.is_symlink()
        if not path.is_file():
            continue
        name = path.relative_to(PACKAGE).as_posix()
        data = path.read_bytes()
        inspect(name, data)
        records[name] = {'bytes': len(data), 'sha256': sha(data)}
    write_json(PACKAGE / 'sha256.json', records)
    result = {'package': PACKAGE.relative_to(ROOT).as_posix(), 'sealed': True,
              'indexed_files': len(records), 'indexed_bytes': sum(r['bytes'] for r in records.values()),
              'index_sha256': sha((PACKAGE / 'sha256.json').read_bytes()),
              'physical_files_including_index': len(records) + 1,
              'physical_bytes_including_index': sum(r['bytes'] for r in records.values()) + (PACKAGE / 'sha256.json').stat().st_size}
    write_json(PERF / 'v622-package-seal.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
