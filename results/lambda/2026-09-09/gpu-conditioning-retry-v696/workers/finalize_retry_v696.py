from pathlib import Path
import hashlib
import json
import shutil

root = Path('results/lambda/2026-09-09/gpu-conditioning-retry-v696')
viewer = Path('results/lambda/2026-09-06/visualiser/data/gtoc12-retry-v696')
(root / 'viewer-dataset').mkdir(exist_ok=True)
for name in ('fleet.json', 'manifest.json', 'compute.json'):
    shutil.copyfile(viewer / name, root / 'viewer-dataset' / name)
for name in ('prepare_retry_viewer_v696.mjs', 'finalize_retry_v696.py', 'verify_retry_staging_v696.py'):
    shutil.copyfile(Path('build/performance') / name, root / 'workers' / name)
(root / 'viewer-validation.json').write_text(json.dumps({
    'dataset': 'gtoc12-retry-v696',
    'url': 'http://127.0.0.1:4173/?dataset=gtoc12-retry-v696&epoch=69807&preset=oblique&z=1',
    'validation': 'npm run check passed all installed datasets after import; CUA accessibility tree and screenshot inspected',
    'selected_label': 'H100 GPU conditioning retry v696',
    'ships': 23, 'deployed': 196, 'collected': 195,
    'displayed_weighted_kg': 12810.136, 'renderer': 'Active WebGL2 / RTX 5090',
    'visible_scene': True, 'browser_console_inspected': False,
    'samples': 11679, 'context_points': 73562,
    'timing_scope': 'Viewer displays internal campaign wall time 133.807 s; summary process time 135.406 s also includes startup.',
}, indent=2) + '\n')
(root / 'README.md').write_text('''# GPU conditioning retry v696

The optional device-controlled first inaccurate-result retry fixes the captured
asteroid 13077 to Earth return without extra attempts or relaxed qualification.
Six complete replays converge and pass independent physics checks on each GPU.
Each GPU passes five new tests and 117 existing regressions. The final reporting
overlay passes the five tests again; H100 targeted pooled graph memcheck reports
zero errors. Wider vendor sanitizer limitations remain.

The paired 225-leg runs preserve 205 qualified legs on each GPU, with no losses.
Solver time falls from 121.3415 to 112.6747 seconds locally and from 155.8527 to
137.6051 seconds on H100. Each is a single paired run, including failed attempts.
The maximum shared final-mass differences are 6.55e-8 and 3.92e-7 kg respectively.

Both new local full campaigns and the H100 campaign converge on all 36 native
legs. The two local baselines fail on the last lower-scoring alternative. Every
retained best fleet passes official and independent full-fleet checkers at
12,810.136 weighted kg / 14,051.855 raw kg: 23 ships, 195 collected asteroids,
196 deployed miners. The score is unchanged. Local process medians are 79.607 s
baseline and 76.157 s candidate, but completed verification work differs. H100
candidate process time is 135.406 s; no matched H100 full baseline is claimed.

## Contents and provenance

- `summary.json`: exact comparison totals and full-fleet qualification.
- `published-source.json`: eight final source hashes and frozen native base.
- `local-raw.tar.gz`, `h100-raw.tar.gz`: full frozen source, prepared QOCO,
  binaries, experiments, logs, vectors, failures and final campaigns.
- Each archive has a member manifest and archive record. Every archived member
  was hash-verified during retrieval/publication.
- `local/`, `h100/`: compact raw reports and selected validation logs.
- `h100-best/Result.txt`: downloaded complete solution, with original viewer
  export and campaign report beside it.
- `viewer-dataset/`: exact three files imported into the existing web viewer.
- `viewer-validation.json`: observed display and import validation.
- `workers/`: original bounded experiment and publication drivers. These record
  machine-specific paths; many use exclusive output directories and are not
  idempotent. Read each driver before adapting or rerunning it.
- `sha256.json`: all package files except the manifest itself.

The native build starts from e7da7d9e plus the five recorded native source files.
The final Python telemetry overlay was validated with the same native binaries.
The final formatted preparation tool reproduces the compiled vendor source
exactly. Concurrent persistent-solver diagnostic changes were excluded from
these frozen builds.

Initial v691/v693 campaigns failed before GPU work because fixture files were
missing. The archives retain those failures, the hash-checked fixture correction,
and successful v694/v695 reruns. Publication checks report success, not merely
the runner exit status. The prior v680 regularization-description correction is
documented separately; its original raw data is unchanged.

The policy remains opt-in (`SPACEPDHCG_TEST_GTOC12_CONDITIONING_RETRY=1`, Ruiz 0).
CUDA controls this numerical retry; Python still controls wider search and
independent CPU checkers certify the physics. No complete GPU-only pipeline or
whole-program sanitizer-clean claim is made.

[Implementation and copy-paste visualiser instructions](../../../../docs/GPU_CONDITIONING_RETRY.md)

Open the running viewer at:
<http://127.0.0.1:4173/?dataset=gtoc12-retry-v696&epoch=69807&preset=oblique&z=1>
''')
records = {}
for path in sorted(root.rglob('*')):
    if path.is_file() and path != root / 'sha256.json':
        raw = path.read_bytes()
        records[path.relative_to(root).as_posix()] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
(root / 'sha256.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps({'package_files': len(records), 'bytes': sum(r['bytes'] for r in records.values())}))
