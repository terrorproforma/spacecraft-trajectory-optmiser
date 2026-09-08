from pathlib import Path
import hashlib,json,shutil
root=Path('results/lambda/2026-09-09/gpu-joint-itinerary-v642')
shutil.copy2('build/performance/publish_joint_v642.py',root/'workers/publish_joint_v642.py')
shutil.copy2('build/performance/finalize_joint_publication_v642.py',root/'workers/finalize_joint_publication_v642.py')
text='''# H100 joint itinerary search and verified fleet replays

See [the project report](../../../../docs/GPU_JOINT_ITINERARY.md) for measured
scope, exact execution flags, GPU/CPU boundaries and visualiser instructions.

`summary.json` records all eight successful full-fleet process timings and
component benchmarks. `RTX5090` and `H100` retain compact original reports.
`h100-best/Result.txt` is the retrieved complete H100 solution; its viewer
export and complete campaign report accompany it. `viewer-dataset` is the
regenerated dataset for the existing web visualiser.

Both `*-raw.tar.gz` archives contain frozen source, tested core/QOCO binaries,
validation and sanitizer logs, all campaign attempts and full exports. Their
`*.manifest.json` files hash each member; `*.record.json` files hash the archives.
The local failed GPU-lock attempt and successful queued retry are retained.

The final Python compatibility guard is separately frozen and tested. Benchmark
and campaign sources remain exactly as run; no later Python version is silently
substituted into their archived provenance. Workers retain the build, test,
benchmark, retrieval and publication scripts with their original machine paths.

The recovered score is 12,810.135953 weighted kg from 14,051.854894 physical kg,
23 ships and 195 collected asteroids. It reproduces the earlier v595 improvement;
sub-microgram replay differences are not new records. Independent CPU physics
verification and Python orchestration remain. Joint batch execution is opt-in.
'''
(root/'README.md').write_text(text)
manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'sha256.json'}
(root/'sha256.json').write_text(json.dumps(manifest,indent=2))
for name,record in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==record['sha256']
print(json.dumps(dict(files=len(manifest),bytes=sum(v['bytes'] for v in manifest.values()))))
