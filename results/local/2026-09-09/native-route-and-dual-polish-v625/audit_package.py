"""Verify a v625 publication and replay saved-data audits only (standard library)."""
from __future__ import annotations
import argparse
import difflib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile

PRIOR_ARCHIVE = '4f43b84567bb9b2076e8e81d6bfe3fb26a5d454a08a8ee5ee3b7f6b368bb1316'
PRIOR_INDEX = '8a0029d2dc9163a7662154f761f5f9339884e8715d8b3a62c1312d3e0a0cfb38'
OMITTED = 'build/performance/seeded-route-v625/output/fleet/official/GTOC12_Verify'
OMITTED_SHA = 'd4e4bc81129266420b27c9bde038bce9eda1960e7de9c695772fbfdb1cc82cd6'
OMITTED_BYTES = 187200
ROUTE_ROOT = 'build/performance/seeded-route-v625'
ROUTE_REVIEW = 'build/performance/seeded-route-review-v625'
H100_ROOT = 'build/performance/native-seed-h100-v625'
PINS = {
    'src/spacepdhcg/gtoc12/fixed_refinement.py': '3850b09e7d146f2e7362b674490a76cec6e113c8b74a655635f16e784bd3b0f1',
    ROUTE_ROOT + '/overlay/spacepdhcg/gtoc12/fixed_refinement.py': '1cc99d88159d4f13e108e4a53ce81c3f74c78e24f24e1b123e0b341116ec35cc',
    ROUTE_ROOT + '/ready.json': '2324d89ee541cde90364c587b68ae15385aa5a4f5e615278eb49a6e8ec30a4b5',
    ROUTE_ROOT + '/output/report.json': '6bdd68e982a8256a87048825d577e5920369413bcc9744cd9feff919abd205f3',
    ROUTE_REVIEW + '/audit.py': '8e2613182fc5d1e6c5a28848973e46b61ce962883e0bd1366d3f375dc6035fb9',
    ROUTE_REVIEW + '/findings-final.json': '9b6e61463c37122ce7e865d83dc49972dec6f3f5752d18f3a7d7e1f25f50bf7c',
    H100_ROOT + '/sha256.json': '4559cc6049501939266bfe7d5cd0b02034070c63d92fd353109d86a3cb6e6cb1',
    H100_ROOT + '/audit_saved.py': '029c12ef080b264a07e2c46f45090e9bce0ecf33f77bc235b8713cb11d9f09dd',
    H100_ROOT + '/findings.json': '22ba730958c05890b69d0d558b28cc5c266d8ca69b2f9fc5489fe86dd55c7a25',
    'build/performance/native-seed-core-v624c/manifest.json': 'eb2c43c8c949b8cc57028b10cb3c116b30a749da0b5c6c1fb811af346d7a0640',
    'build/performance/native-seed-core-v624c/source.tar.gz': 'bbc7146bd448a7639c7dbf2d9100f8fc74e75e1f21507d3da159308cfd8bb375',
    'build/performance/dual-polish-v625/final-audit.json': '1810a166c0901618f68abc3cdc8591d18a6942caac93ecf82e55bfb177227487',
    'build/performance/dual-polish-review-v625/findings.json': '69a3e3ad1fc21b32ee49353427cdac37987facabc3a500d2ef7e6f87aa9f74d4',
}

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha(path):
    return sha_bytes(path.read_bytes())

def read(path):
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def safe_name(name):
    p = PurePosixPath(name)
    assert name and not p.is_absolute() and not any(x in {'.', '..'} for x in p.parts), name
    assert p.parts and name == p.as_posix(), 'noncanonical path: ' + name
    assert '\\' not in name and ':' not in name and '\0' not in name, name
    assert not any(x in {'.git', '__pycache__', '.pytest_cache', '.ruff_cache'} for x in p.parts), name
    assert all(not x.endswith((' ', '.')) for x in p.parts), name
    assert p.suffix.lower() not in {'.pem', '.key', '.so', '.dll', '.exe', '.o', '.obj', '.a', '.pyc'}, name
    return p

class Scanner:
    def __init__(self):
        self.bytes = 0
        self.members = 0
        self.nested_archives = 0

    def payload(self, name, data, depth=0):
        safe_name(name)
        assert depth <= 5
        self.bytes += len(data)
        self.members += 1
        assert self.bytes <= 768 * 1024 * 1024, 'expanded audit size budget'
        assert not data.startswith((b'\x7fELF', b'MZ')), name
        assert not re.search(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----', data), name
        lower = name.lower()
        if lower.endswith(('.zip', '.npz')):
            self.nested_archives += 1
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                info = z.infolist()
                assert len(info) == len({x.filename for x in info})
                for member in info:
                    safe_name(member.filename.rstrip('/'))
                    assert not stat.S_ISLNK(member.external_attr >> 16)
                    assert not (member.flag_bits & 1), 'encrypted nested member'
                    if not member.is_dir():
                        self.payload(member.filename, z.read(member), depth+1)
        elif lower.endswith(('.tar.gz', '.tgz', '.tar')):
            self.nested_archives += 1
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as t:
                members = t.getmembers()
                assert len(members) == len({x.name for x in members})
                for member in members:
                    safe_name(member.name.rstrip('/'))
                    assert member.isfile() or member.isdir(), 'special/link tar member'
                    if member.isfile():
                        self.payload(member.name, t.extractfile(member).read(), depth+1)

def verify_extract(package, target, expected_index, expected_archive, scanner):
    index_path = package / 'index.json'
    assert sha(index_path) == expected_index, 'index identity'
    index = read(index_path)
    archive_path = package / 'evidence.zip'
    assert sha(archive_path) == index['archive_sha256'] == expected_archive
    entries = index['files']
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        assert len(members) == len({m.filename for m in members}) == len(entries)
        assert set(entries) == {m.filename for m in members}
        for member in members:
            assert not member.is_dir() and not stat.S_ISLNK(member.external_attr >> 16)
            assert not (member.flag_bits & 1)
            path = safe_name(member.filename)
            payload = archive.read(member)
            spec = entries[member.filename]
            assert len(payload) == spec['bytes'] and sha_bytes(payload) == spec['sha256'], member.filename
            scanner.payload(member.filename, payload)
            dest = target.joinpath(*path.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
    if 'file_count' in index:
        assert index['file_count'] == len(entries)
    if 'uncompressed_bytes' in index:
        assert index['uncompressed_bytes'] == sum(x['bytes'] for x in entries.values())
    if 'archive_bytes' in index:
        assert index['archive_bytes'] == archive_path.stat().st_size
    return index

def replay(script, argv, destination, label):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void', PYTHONDONTWRITEBYTECODE='1')
    # The two script hashes are checked before executing these saved-data-only helpers.
    completed = subprocess.run([sys.executable, '-B', str(script), *map(str,argv)], env=env,
                               capture_output=True, text=True, timeout=60)
    (destination / (label+'.stdout')).write_text(completed.stdout, encoding='utf-8')
    (destination / (label+'.stderr')).write_text(completed.stderr, encoding='utf-8')
    assert completed.returncode == 0, label + ': ' + completed.stderr[-3000:]
    return {'exit_code': completed.returncode, 'script_sha256': sha(script)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--expected-index-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    assert re.fullmatch('[0-9a-f]{64}', args.expected_index_sha256)
    assert not args.output_dir.exists(), 'fresh review output required'
    args.output_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='route-publication-v625-') as name:
        root = Path(name)
        scanner = Scanner()
        old = verify_extract(args.previous, root, PRIOR_INDEX, PRIOR_ARCHIVE, scanner)
        # Caller-supplied index digest binds the complete current archive digest/map.
        assert sha(args.package / 'index.json') == args.expected_index_sha256
        archive_digest = read(args.package / 'index.json')['archive_sha256']
        index = verify_extract(args.package, root, args.expected_index_sha256, archive_digest, scanner)
        assert index['previous_evidence']['sha256'] == PRIOR_ARCHIVE
        omitted = [x for x in index['excluded'] if x.get('reason') == 'executable']
        assert omitted == [{'path':OMITTED, 'reason':'executable', 'sha256':OMITTED_SHA, 'bytes':OMITTED_BYTES}]
        assert OMITTED not in index['files'] and not (root / OMITTED).exists()
        for name, expected in PINS.items():
            assert sha(root / name) == expected, name
        cpu = read(root / 'build/performance/seeded-hooks-python-v625/final-report.json')
        assert cpu['GPU_calls'] == 0 and cpu['pytest']['passed'] == 73 and cpu['pytest']['exit_code'] == cpu['ruff']['exit_code'] == 0
        for name, expected in cpu['source_sha256'].items():
            assert sha(root / name) == expected
        assert cpu['measured_route_hook_sha256'] == PINS[ROUTE_ROOT+'/overlay/spacepdhcg/gtoc12/fixed_refinement.py']
        assert sha(root / cpu['measurement_source']) == cpu['measured_route_hook_sha256']
        difference = '\n'.join(difflib.unified_diff(
            (root/cpu['measurement_source']).read_text().splitlines(),
            (root/'src/spacepdhcg/gtoc12/fixed_refinement.py').read_text().splitlines(),
            fromfile='measured-hook-1cc99d88', tofile='published-reporting-hook-3850b09e'))+'\n'
        (args.output_dir/'reporting-only.diff').write_text(difference)
        h100_saved = read(root / H100_ROOT / 'findings.json')
        h100_output = args.output_dir.resolve() / 'h100-replayed.json'
        h100 = replay(root/H100_ROOT/'audit_saved.py', ['--root',root/H100_ROOT,'--output',h100_output], args.output_dir, 'h100')
        assert read(h100_output) == h100_saved, 'H100 physical/provenance findings changed'
        route_saved = read(root / ROUTE_REVIEW / 'findings-final.json')
        route_output = args.output_dir.resolve() / 'route-replayed.json'
        route = replay(root/ROUTE_REVIEW/'audit.py', ['--root',root,'--output',route_output], args.output_dir, 'route')
        route_replayed = read(route_output)
        original_outputs = route_saved['actual']['output_files'].copy()
        key = 'fleet/official/GTOC12_Verify'
        assert original_outputs.pop(key) == OMITTED_SHA
        assert route_replayed['actual']['output_files'] == original_outputs
        expected_route = json.loads(json.dumps(route_saved))
        expected_route['actual']['output_files'] = original_outputs
        assert route_replayed == expected_route, 'route physical/source/count findings changed'
        dual = read(root/'build/performance/dual-polish-v625/final-audit.json')
        exact = read(root/'build/performance/dual-polish-review-v625/findings.json')
        assert dual['hybrid_candidate_pass'] and dual['original_native_termination_unchanged']
        assert exact['passed'] and exact['native_termination'] == 2 and not exact['native_qualified']
        assert exact['optimizer_calls'] == exact['GPU_calls'] == exact['propagation_calls'] == 0
        route_result = read(root/ROUTE_ROOT/'output/report.json')
        assert route_result['native_solves_started'] == route_result['native_iterations'] == 19
        assert route_result['flight_certificate_calls'] == 19 and route_result['wait_certificate_calls'] == 2
        assert route_result['regression']['regression_passed'] and not route_result['regression']['incumbent_promoted']
        overlap = {name: {'previous':old['files'][name]['sha256'],'current':spec['sha256']}
                   for name,spec in index['files'].items() if name in old['files'] and old['files'][name] != spec}
        finding = {
            'passed':True, 'package_index_sha256':args.expected_index_sha256, 'archive_sha256':archive_digest,
            'previous_archive_sha256':PRIOR_ARCHIVE, 'previous_index_sha256':PRIOR_INDEX,
            'current_indexed_files':len(index['files']), 'previous_indexed_files':len(old['files']),
            'current_indexed_bytes':sum(x['bytes'] for x in index['files'].values()),
            'archive_safety': {'members_checked_including_nested':scanner.members,'bytes_checked_including_nested':scanner.bytes,'nested_archives_checked':scanner.nested_archives,'no_executables_keys_git_caches_or_links':True},
            'ordered_unpack':'Exact prior v624 archive first, indexed v625 overlays second, fresh temporary root.',
            'changed_prior_paths':overlap, 'source_pins':PINS,
            'production_reporting_source_distinct_from_measured_source':True,
            'CPU_regressions_saved_passed':73, 'h100_saved_replay':h100, 'route_saved_replay':route,
            'omissions': [{'path':OMITTED,'sha256':OMITTED_SHA,'bytes':OMITTED_BYTES,
                          'binary_bytes_reverified':False,'statement':'Compiled official checker omitted; recorded invocation, identity and results retained. No checker was executed by this audit.'}],
            'route_output_hash_exception_is_exactly_declared_binary':True,
            'all_H100_findings_equal':True, 'all_route_physical_findings_equal':True,
            'dual_saved_numeric_pass_native_limit_preserved':True,
            'new_GPU_calls':0,'new_optimizer_calls':0,'new_propagations':0,'new_official_checker_calls':0,
        }
    (args.output_dir/'findings.json').write_text(json.dumps(finding,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'findings':str(args.output_dir/'findings.json'),'sha256':sha(args.output_dir/'findings.json')}))

if __name__ == '__main__':
    main()
