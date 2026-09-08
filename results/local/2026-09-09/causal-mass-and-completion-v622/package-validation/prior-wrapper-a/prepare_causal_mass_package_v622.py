"""CPU-only draft inventory for v622. Never seals, publishes, runs Git or CUDA."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import re
import tarfile


ROOT = Path(__file__).resolve().parents[2]
PERF = 'build/performance/'
TARGET = 'results/local/2026-09-09/causal-mass-and-completion-v622'
DENY_PARTS = {'.git', '.ssh', '__pycache__', '.pytest_cache', '.ruff_cache', 'pytest-tmp'}
DENY_SUFFIXES = {'.so', '.dll', '.exe', '.o', '.obj', '.a', '.lib', '.pyc', '.pyd', '.pem', '.key', '.p12', '.pfx'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative(value):
    path = PurePosixPath(value.replace('\\', '/'))
    assert not path.is_absolute() and '..' not in path.parts and path.parts
    assert ':' not in str(path), 'Absolute/private path cannot become a package member'
    return path.as_posix()


def safe_bytes(name, data):
    path = PurePosixPath(relative(name))
    assert not any(p in DENY_PARTS for p in path.parts), name
    assert path.suffix.lower() not in DENY_SUFFIXES, name
    assert not data.startswith((b'\x7fELF', b'MZ')), 'Executable payload: ' + name
    assert not re.search(br'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----', data), 'Private-key payload: ' + name


def components():
    result = [
        ('mass_build_a', 'mass-core-v622a', 'mass/build/a', 'manifest.json'),
        ('mass_tiny', 'mass-tiny-v622', 'mass/tiny', 'report.json'),
        ('mass_preflight', 'mass-real-v622-preflight', 'mass/preflight', 'run/report.json'),
        ('mass_oracle', 'mass-native-oracle-v622', 'reviews/mass-oracle', 'fixtures.json'),
        ('mass_review', 'mass-native-review-v622', 'reviews/mass-native', 'tiny-findings.json'),
        ('mass_real', 'mass-real-v622', 'mass/real', 'run/report.json'),
        ('mass_real_review', 'mass-real-review-v622', 'reviews/mass-real', 'findings.json'),
        ('admission', 'refinement-admission-v622', 'admission', 'ready.json'),
        ('compact_review', 'completion-model-review-v622', 'reviews/compact', 'readback-findings.json'),
        ('benchmark', 'completion-pack-benchmark-v622', 'completion/benchmark', 'ready.json'),
    ]
    result += [(f'compact_build_{v}', f'completion-model-v622{v}', f'completion/build/{v}', 'report.json')
               for v in 'abcdefg']
    result += [(f'compact_gpu_{v or "initial"}', f'completion-model-gpu-v622{v}',
                f'completion/runs/{v or "initial"}', 'report.json') for v in ('', 'b', 'c', 'd')]
    return [{'name': name, 'source': PERF + source, 'destination': destination,
             'anchor': anchor, 'root_terminal_pin': None} for name, source, destination, anchor in result]


DEPENDENCIES = [
    PERF + 'run_mass_tiny_v622.py', PERF + 'run_mass_real_v622.py',
    PERF + 'generate_mass_fixture_v622.py', PERF + 'check_mass_build_v622.py',
    PERF + 'l1-weight-real-review-v618/decimal_audit_source.py',
    PERF + 'l1-weight-real-review-v618/review_real.py',
    PERF + 'completion-integration-review-v621/review_native_readback.py',
    PERF + 'completion-fullcore-v621/source.tar.gz',
    PERF + 'completion-fullcore-v621/manifest.json',
    PERF + 'completion-fullcore-v621/resource-usage.log',
    PERF + 'l1-weight-v618b/source.tar.gz',
    PERF + 'l1-weight-v618b/manifest.json',
]


def inventory(plan, output):
    output.mkdir(parents=True, exist_ok=False)
    artifacts, snapshots, archives, objects, statuses = {}, {}, {}, {}, []

    def object_row(name, data):
        safe_bytes(name, data)
        row = {'bytes': len(data), 'sha256': digest(data)}
        old = objects.setdefault(row['sha256'], row['bytes'])
        assert old == row['bytes']
        return row

    def source_archive(source_rel, data):
        archive_sha = digest(data)
        if archive_sha in archives:
            archives[archive_sha]['original_paths'].append(source_rel)
            return
        members = {}
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as archive:
            for item in archive:
                assert item.isfile() or item.isdir(), 'Archive links/special members rejected'
                name = relative(item.name)
                if item.isdir():
                    continue
                assert name not in members
                payload = archive.extractfile(item).read()
                members[name] = object_row(name, payload)
        archives[archive_sha] = {'bytes': len(data), 'original_paths': [source_rel], 'members': members}

    def scan_file(source, destination, source_group=None, source_member=None):
        data = source.read_bytes()
        source_rel = source.relative_to(ROOT).as_posix()
        safe_bytes(source_rel, data)
        if source_group is not None:
            snapshots.setdefault(source_group, {})[relative(source_member)] = object_row(source_member, data)
        else:
            assert destination not in artifacts, destination
            artifacts[destination] = {'origin': source_rel, 'bytes': len(data), 'sha256': digest(data)}
            if source.name.endswith(('.tar.gz', '.tar')):
                source_archive(source_rel, data)

    for component in plan['components']:
        source_rel = relative(component['source'])
        source = ROOT / source_rel
        anchor = source / relative(component['anchor'])
        status = {'name': component['name'], 'present': source.exists(), 'anchor_present': anchor.is_file(),
                  'root_terminal_pin': component.get('root_terminal_pin')}
        if anchor.is_file():
            data = anchor.read_bytes()
            status['anchor_sha256'] = digest(data)
            parsed = json.loads(data)
            status['recorded_complete'] = parsed.get('complete')
            status['recorded_status'] = parsed.get('status')
            pin = component.get('root_terminal_pin')
            if pin:
                assert pin['sha256'] == digest(data), component['name']
        statuses.append(status)
        if not source.is_dir():
            continue
        for path in sorted(source.rglob('*')):
            assert not path.is_symlink(), path
            if not path.is_file():
                continue
            rel = path.relative_to(source)
            if any(part in DENY_PARTS for part in rel.parts):
                continue
            # Source revisions remain distinct while common file bytes share objects.
            if rel.parts[0] in ('source', 'implementation'):
                group = source_rel + '/' + rel.parts[0]
                scan_file(path, None, group, Path(*rel.parts[1:]).as_posix())
            else:
                scan_file(path, relative(component['destination']) + '/' + rel.as_posix())
    for source_rel in plan['dependencies']:
        source_rel = relative(source_rel)
        source = ROOT / source_rel
        if not source.is_file():
            statuses.append({'dependency': source_rel, 'present': False})
            continue
        scan_file(source, 'dependencies/' + source_rel.removeprefix(PERF))

    # Reconstruct the exact baseline source inventory offline from saved source bytes.
    # The original review already bound this map to Git; this check invokes no Git.
    mass_manifest = ROOT / (PERF + 'mass-core-v622a/manifest.json')
    baseline = json.loads(mass_manifest.read_text())['base_source_sha256']
    missing_baseline = {name: value for name, value in baseline.items() if value not in objects}
    report = {
        'state': 'draft_inventory_only', 'sealed': False, 'gpu_calls': 0, 'git_calls': 0,
        'target': TARGET, 'components': statuses, 'artifacts': artifacts,
        'source_snapshots': snapshots, 'source_archives': archives,
        'source_objects': {key: {'bytes': value} for key, value in sorted(objects.items())},
        'baseline_source_objects_missing': missing_baseline,
        'deduplication': {
            'expanded_source_files': sum(len(v) for v in snapshots.values()),
            'archive_source_members': sum(len(v['members']) for v in archives.values()),
            'unique_source_objects': len(objects), 'unique_source_bytes': sum(objects.values()),
            'unique_original_archives': len(archives),
            'policy': 'Keep each original source archive once per SHA to preserve original container identity; materialized source/implementation revisions use one content-addressed source store and exact per-revision maps.',
        },
        'requirements_before_final_assembly': [
            'Root terminal pins for mass real, compact final and bounded benchmark, plus their final reviews.',
            'Preserve early compact failures/partial readbacks/zero-call busy attempts as separate components.',
            'Keep 34 fresh queue controls, 54 total saved readbacks and later 76-test integration revisions distinct.',
            'Materialize into a fresh repository-relative CPU tree; source logs remain byte-exact.',
            'Sequential raw-vector/array/identity checks only; never import a native library or evaluate a trajectory.',
            'Only after terminal evidence and portable checks: final README, archive member maps and sha256 index.',
        ],
    }
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (output / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'draft': True, 'sealed': False, 'artifacts': len(artifacts),
                      'unique_source_objects': len(objects), 'missing_base_objects': len(missing_baseline),
                      'output': output.relative_to(ROOT).as_posix()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text()) if args.plan else {
        'target': TARGET, 'sealed': False, 'components': components(), 'dependencies': DEPENDENCIES,
        'source_semantics': 'No final assembly, index, Git operation, upload or GPU launch is implemented by this preparer.',
    }
    output = args.out.resolve()
    output.relative_to(ROOT)
    inventory(plan, output)


if __name__ == '__main__':
    main()
