from pathlib import Path
import hashlib,json,shutil
p=Path('build/performance');target=Path('results/lambda/2026-09-08/gpu-fast-lambert-v589')
summary=json.loads((target/'summary.json').read_text())
viewer=Path('results/lambda/2026-09-06/visualiser/data/gtoc12-v588')
manifest=json.loads((viewer/'manifest.json').read_text())
assert hashlib.sha256((viewer/'fleet.json').read_bytes()).hexdigest()==manifest['files']['fleet.json']['sha256']
assert manifest['summary']['ships']==1 and manifest['summary']['unique_asteroids']==8
summary['viewer']=dict(dataset='gtoc12-v588',source='campaign-final/default/output/fleet',weighted_score_kg=548.2546201231003,replay_points=manifest['summary']['replay_points'],context_points=manifest['kepler_check']['context_points_checked'],url='http://127.0.0.1:4173/?dataset=gtoc12-v588&epoch=69807&preset=oblique&z=1',files_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in viewer.iterdir() if f.is_file()})
(target/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ['prepare_fast_root_viewer.py','fast-root-viewer-fleet.json','finalize_fast_root_v589.py']:
 shutil.copy2(p/name,target/name)
(target/'README.md').write_text('''# CUDA Lambert root measurements, 8 September 2026

`summary.json` compares the complete A/B/B/A campaign and isolated screening on
RTX 5090 and Lambda H100, with final default-build validation. See
`docs/GPU_FAST_LAMBERT_ROOTS.md` in the repository for method and limitations.

`lambda-raw.tar.gz` contains 161 verified source/runtime/result files, including
the final H100 trajectory at `campaign-final/default/output/fleet/Result.txt`.
`local-raw.tar.gz` contains local runs, both profiling attempts, the original
empty-fixture test failure, frozen binaries and reproduction workers.
The archive member manifests verify every file. `retrieval.json` records the
remote archive checksum; `summary.json` records the local archive checksum.
`files-sha256.json` covers published files. Original experiment binaries have the
new method opt-in; final binaries enable it by default. Exact source is archived.

All ten campaigns pass official and independent mission verification. These
one-ship runs retain 548.254620 weighted kg; no new fleet record is claimed.
The best fleet remains 12,805.194 weighted kg. Lambert sanitizer passes do not
resolve the separately documented vendor full-solver sanitizer failures.

The imported final H100 trajectory is displayed in the existing local viewer:
http://127.0.0.1:4173/?dataset=gtoc12-v588&epoch=69807&preset=oblique&z=1
Dataset files live in `results/lambda/2026-09-06/visualiser/data/gtoc12-v588`.
''')
files={f.relative_to(target).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(target.rglob('*')) if f.is_file() and f.name!='files-sha256.json'}
(target/'files-sha256.json').write_text(json.dumps(files,indent=2))
print(json.dumps(dict(published_files=len(files),viewer=summary['viewer'],local=summary['local_archive'])))
