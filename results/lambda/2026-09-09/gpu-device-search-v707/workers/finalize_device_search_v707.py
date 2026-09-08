from pathlib import Path
import hashlib
import json
import shutil

root=Path('results/lambda/2026-09-09/gpu-device-search-v707')
viewer=Path('results/lambda/2026-09-06/visualiser/data/gtoc12-search-v707')
(root/'viewer-dataset').mkdir(exist_ok=True)
for name in ('fleet.json','manifest.json','compute.json'):shutil.copyfile(viewer/name,root/'viewer-dataset'/name)
for name in ('prepare_device_search_v707.mjs','finalize_device_search_v707.py','verify_device_search_staging_v707.py'):
    shutil.copyfile(Path('build/performance')/name,root/'workers'/name)
(root/'viewer-validation.json').write_text(json.dumps(dict(dataset='gtoc12-search-v707',url='http://127.0.0.1:4173/?dataset=gtoc12-search-v707&epoch=69807&preset=oblique&z=1',selected_label='H100 GPU epoch search v707',ships=23,deployed_asteroids=196,collected_asteroids=195,displayed_weighted_kg=12810.136,renderer='Active WebGL2 / RTX 5090',visible_scene=True,validation='npm run check passed all installed datasets; CUA accessibility tree and screenshot inspected',samples=11679,context_points=73562,asteroid_crosscheck_max_km=3.59e-6,earth_crosscheck_max_km=3.07e-7,browser_console_inspected=False,timing_scope='Viewer internal campaign wall time 133.099 s; process total 134.643 s includes startup.'),indent=2)+'\n')
(root/'README.md').write_text('''# GPU-controlled epoch search v707

One conditional CUDA graph now controls the complete fixed-order epoch search:
initial evaluation, neighbourhood generation, geometry, forward-mass evaluation,
winner acceptance, mesh transitions and deadlines. The final candidate retains
exactly matching search results in repeated paired comparisons.

The four warm paired cases are 1.37-1.60x faster on RTX 5090 and 2.29-2.78x on
H100. These are complete epoch-search stage timings including setup, transfers
and report construction, not full-mission speedups or certified solutions/s.
Both GPUs pass 111 tests without skips. The final moving-search test passes
racecheck and synccheck on both, and memcheck on H100. Local active-loop
memcheck returns CUDA error 999; empty-loop memcheck passes. This limitation
remains unresolved and there is no whole-program sanitizer-clean claim.

All four complete fleet runs (baseline/candidate on each GPU) converge on all
36 native legs and pass both official and independent full-fleet checkers.
Each evaluates 18 orders and 23,142 joint candidates. Four device searches
accept 112 moves in 128 neighbourhoods; joint host calls fall from 132 to 4,
and epoch transfers from 39,888 to 1,248 bytes each way. The score remains
12,810.136 weighted kg / 14,051.855 raw kg: 23 ships, 195 collected asteroids,
196 deployed miners. Local process time is 76.936 vs 75.867 seconds and H100
135.002 vs 134.643 seconds. These single pairs establish no reliable additional
whole-campaign speedup.

## Evidence layout

- `summary.json`: exact paired stage medians, full campaigns, qualification,
  work counts, runtime hashes and the instrumentation limitation.
- `published-source.json`: the five final source files, identical on both GPUs,
  frozen from commit 3175fb44 plus this implementation. Concurrent persistent
  backend changes were excluded from the measured builds.
- `local-raw.tar.gz`, `h100-raw.tar.gz`: complete frozen source, core binaries,
  reused QOCO binary, tests, raw timing samples, failures and full campaigns.
  Each has an archive record and a hash/length manifest for every member.
  All members were checked on retrieval. QOCO's prepared source is retained in
  the preceding `gpu-conditioning-retry-v696` evidence package.
- `local/`, `h100/`: compact copies of final validation logs, follow-up benchmark
  results, campaign reports and explicit campaign launch gates.
- `h100-best/Result.txt`: downloaded full H100 solution, with its original
  trajectory export and complete campaign report beside it.
- `viewer-dataset/`: exact imported files used by the web visualiser.
- `viewer-validation.json`: observed UI, renderer and import checks.
- `workers/`: bounded experiment, retrieval and publication drivers. They record
  machine-specific paths and many use exclusive output directories. Read before
  adapting; they are not generally idempotent. Campaigns use `worker-v706.py`
  from the raw campaign archive, whose hash/gates are recorded in `launch.json`.
- `sha256.json`: hashes and sizes of every package file except itself.

## Failures and scope

The first controller's block reducer failed synchronization checking inside the
loop on both GPUs. The final full-warp reducer passes that check. Raw archives
retain both sources, binaries and logs; their results must not be conflated.

The first H100 configure failed because its isolated source lacked the Git
provenance required by CMake. The successful build initialized that repository.
The first H100 benchmark failed before GPU execution because an editable import
hook hid the frozen package; the follow-up removes it and succeeds using the
same source and binary. Initial local regression skipped seven archived-route
tests; the final 111-test runs include that fixture.

Accordingly, the original final-worker report records `success: false` for the
local memcheck limitation or H100 benchmark-launch failure. The separate
follow-up reports, sanitizer logs and explicit campaign launch gates describe
what subsequently passed. Failures have not been rewritten as successes.

The feature remains opt-in through `SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SEARCH=1`
with the existing CUDA joint backend active. Python still controls wider route
and fleet orchestration. Independent CPU physics checks and file output remain
explicit end-to-end costs; this is not completion of the whole GPU-native goal.

[Implementation and copy-paste visualiser instructions](../../../../docs/GPU_DEVICE_EPOCH_SEARCH.md)

<http://127.0.0.1:4173/?dataset=gtoc12-search-v707&epoch=69807&preset=oblique&z=1>
''')
manifest={}
for p in sorted(root.rglob('*')):
    if p.is_file() and p!=root/'sha256.json':
        raw=p.read_bytes();manifest[p.relative_to(root).as_posix()]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(root/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(files=len(manifest),bytes=sum(r['bytes'] for r in manifest.values()))))
