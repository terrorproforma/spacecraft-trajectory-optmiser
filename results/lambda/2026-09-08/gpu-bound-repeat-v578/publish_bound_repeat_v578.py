from pathlib import Path
import hashlib,json,shutil,statistics,tarfile
p=Path('build/performance');root=p/'retrieved-bound-repeat-v578'
target=Path('results/lambda/2026-09-08/gpu-bound-repeat-v578')
analysis=json.loads((root/'analysis.json').read_text())
first=p/'retrieved-bound-types-v577/validation-v569'
reports=[json.loads((path/'report.json').read_text()) for path in [first,root]]
assert reports[0]['runtime_sha256']==reports[1]['runtime_sha256']
assert analysis['source_core_sha256'] in reports[0]['runtime_sha256'].values()
for name,sha in reports[0]['source_sha256'].items():
 if name.startswith(('cpp/','src/','tests/','scripts/')):assert reports[1]['source_sha256'][name]==sha,name
for mode in ['baseline','candidate']:
 for i,path in enumerate([first,root]):
  rows=json.loads((path/mode/'results.json').read_text())
  assert len(rows)==225 and all(x['certified'] for x in rows if x['status']=='converged')
  assert sum(x['status']=='converged' for x in rows)==205
  assert sum(x['seconds'] for x in rows)==analysis['seconds'][mode][i]
shutil.copy2(root/'analysis.json',target/'summary.json')
for name in ['archive_bound_repeat_v578.py','collect_bound_repeat_v578.py','prepare_bound_types_repeat_v575.py','publish_bound_repeat_v578.py']:
 shutil.copy2(p/name,target/name)
(target/'README.md').write_text('''# Repeated H100 bound-classification comparison

This supplements `../gpu-bound-types-v577` with a second full 225-leg pass per
mode on the same frozen binary. The initial pair ran host then GPU classification;
the repeat ran GPU then host, after the intervening campaign/default validation.
Both modes use device numerical initialization, zero Ruiz and workspace reuse.
Original solver budgets and physics tolerances are unchanged.

All four passes retain the same 205 independently certified trajectories. The
maximum final-mass difference relative to the first baseline is 1.042e-7 kg.
Host solver times are 157.0263 and 154.5054 seconds; GPU times are 164.3139 and
153.9951 seconds. The two-run medians are 155.7659 and 159.1545 seconds: GPU
classification takes 2.18% more time in this sample. Timing ranges overlap.
The repeated pair alone slightly favours GPU classification, illustrating the
variation. This does not establish either an overall speedup or its cause.

`lambda-raw.tar.gz` contains the second pair, source snapshots, exact commands,
per-leg trajectories and certificates. First-pair raw data and the actual CUDA
binaries are retained in `../gpu-bound-types-v577/lambda-raw.tar.gz`. Runtime and
source identities were checked across both pairs. `summary.json` summarizes all
four passes. Member and published-file SHA-256 manifests accompany this archive.
''')
(target/'.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
manifest={path.relative_to(target).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(target.rglob('*')) if path.is_file() and path.name!='files-sha256.json'}
(target/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(analysis,indent=2))
