"""Audit saved frozen build/source evidence; stdlib only, no compiler/native execution."""
from pathlib import Path
import hashlib
import io
import json
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BUILD = ROOT/'build/performance/mass-core-v622a'
sha = lambda b: hashlib.sha256(b).hexdigest()
raw = (BUILD/'manifest.json').read_bytes()
assert sha(raw) == 'ca970d7acd9893ba71d7e58f85393cb2a26c003ca931e2071e2718e4292ad026'
m = json.loads(raw)
assert m['complete'] and m['gpu_calls'] == 0 and len(m['stages']) == 16
assert all(stage['returncode'] == stage['expected'] for stage in m['stages'])
assert sha((BUILD/'source.tar.gz').read_bytes()) == m['source_archive_sha256']
with tarfile.open(BUILD/'source.tar.gz') as archive:
    assert set(archive.getnames()) == set(m['source_sha256'])
    for name, digest in m['source_sha256'].items():
        assert sha(archive.extractfile(name).read()) == digest, name
prior = json.loads((OUT/'findings.json').read_text())
assert all(prior['source_sha256'][name] == m['owned_sha256'][name] for name in m['owned_paths'])
base_bytes = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'archive', '--format=tar',
                                     m['base_commit'], 'cpp', 'third_party'], cwd=ROOT)
with tarfile.open(fileobj=io.BytesIO(base_bytes), mode='r:') as archive:
    base = {member.name: sha(archive.extractfile(member).read()) for member in archive if member.isfile()}
assert base == m['base_source_sha256']
assert all(m['source_sha256'][name] == value for name, value in base.items() if name not in m['owned_paths'])
assert set(m['source_sha256']) - set(base) == set(m['owned_paths']) - set(base)
assert sha(''.join(name+':'+value+'\n' for name, value in m['source_sha256'].items()).encode()) == m['source_tree_sha256']
text = (BUILD/'resources.log').read_text()
resources = {}
patterns = ['cooperative_l1_kernelILb1', 'cooperative_l1_kernelILb0',
            'cooperative_solve_kernelILb1', 'cooperative_solve_kernelILb0',
            '12solve_kernelILb1', '12solve_kernelILb0', 'cooperative_halpern_kernel',
            'cooperative_mass_kernel', 'cooperative_mass_initialise_kernel']
for name, row in re.findall(r'Function ([^\n]+):\n\s+(REG:[^\n]+)', text):
    keys = [key for key in patterns if key in name]
    if keys:
        assert len(keys) == 1 and keys[0] not in resources
        resources[keys[0]] = {k: int(v) for k, v in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)', row)}
expected = [(94,0,7168),(94,0,7168),(96,0,7168),(80,0,7168),
            (204,40,1032),(148,40,1032),(96,0,7168),(198,40,7168),(48,0,7168)]
for key, values in zip(patterns, expected, strict=True):
    assert tuple(resources[key][part] for part in ('REG','STACK','SHARED')) == values
    assert resources[key]['LOCAL'] == 0
parsers = {}
for case in ('conditioning', 'difficult'):
    for mode in ('unit', 'mass', 'mass-seed'):
        path = BUILD/f'validate-{case}-{mode}.log'
        rows = [json.loads(line.split(' ',1)[1], parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                for line in path.read_text().splitlines() if line.startswith('PERSISTENT_')]
        assert rows and rows[0]['source_tree_sha256'] == m['source_tree_sha256']
        if mode != 'unit':
            assert len(rows[0]['mass_map']) == (211 if case == 'conditioning' else 234)
        parsers[f'{case}-{mode}'] = {'records':len(rows), 'log_sha256':sha(path.read_bytes())}
report = {
    'scope':'Saved CPU/build/source evidence only; no GPU calls or native loading by reviewer.',
    'manifest_sha256':sha(raw), 'source_tree_sha256':m['source_tree_sha256'],
    'source_archive_sha256':m['source_archive_sha256'], 'source_files':len(m['source_sha256']),
    'git_base_verified':m['base_commit'], 'only_reviewed_overlays':True,
    'core':m['library'], 'test':m['test'], 'replay':m['replay'],
    'resources':resources, 'hidden_replay_records':parsers,
    'stage_logs_sha256':{stage['name']:sha((BUILD/(stage['name']+'.log')).read_bytes()) for stage in m['stages']},
    'decision':'Frozen build/source/resource checks pass; tiny launch separately gated by reviewed supervisor.',
}
(OUT/'build-findings.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'source_files':len(m['source_sha256']),'report_sha256':sha((OUT/'build-findings.json').read_bytes())}))
