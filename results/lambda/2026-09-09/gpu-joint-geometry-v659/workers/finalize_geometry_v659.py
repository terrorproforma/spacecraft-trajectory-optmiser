from pathlib import Path
import hashlib,json,shutil,difflib,subprocess
root=Path('results/lambda/2026-09-09/gpu-joint-geometry-v659')
viewer=Path('results/lambda/2026-09-06/visualiser/data/gtoc12-geometry-v659')
shutil.copytree(viewer,root/'viewer-dataset')
for name in ('prepare_geometry_viewer_v659.mjs','audit_geometry_source_v660.py','audit_geometry_proxy_v660.py','prepare_geometry_archive_lambda_v659.py','finalize_geometry_v659.py'):
 shutil.copy2(Path('build/performance')/name,root/'workers'/name)
(root/'README.md').write_text('''# Resident CUDA joint geometry: RTX 5090 and H100

See [the implementation and measurements](../../../../docs/GPU_JOINT_GEOMETRY.md)
for execution flags, exact scope, remaining CPU work and visualiser loading.

`summary.json` contains four component cases per GPU and eight full-fleet recovery
processes. All fleet outputs pass independent and official physics checkers at
unchanged tolerances. The score reproduces 12,810.135953 weighted kg and
14,051.854894 physical kg with 23 ships, 195 collected asteroids and 196 deployed
miners. Component gains are 2.13–4.77x on RTX 5090 and 3.20–6.40x on H100, versus
the preceding staged batched CUDA wrapper. Full-process medians are effectively
unchanged; no overall speedup or score improvement is claimed.

Both GPUs pass 71 tests, with nine resident-geometry tests passing memcheck,
synccheck and racecheck. Each raw archive preserves the frozen source, fixtures,
core and QOCO binaries, commands, all test logs, timing samples and all four fleet
attempts. Archive records and member manifests verify every downloaded byte.
The local original 69-test run and later two additional input-gate tests are
retained separately. `published-source.json` records the subsequent documentation
comment change in the public header; runtime source is otherwise byte-identical.

`h100-best/Result.txt` is the complete downloaded H100 resident-geometry replay.
Its exported histories and campaign report accompany it. `viewer-dataset` is the
validated dataset displayed at:

http://127.0.0.1:4173/?dataset=gtoc12-geometry-v659&epoch=69807&preset=oblique&z=1

The test feature is opt-in. Python still creates moves, packs metadata and drives
the search, and independent CPU verification gates fleet acceptance. Computed
joint geometry costs currently recompute instead of populating the Python cache.
Sanitizer claims apply to resident geometry tests, not all QOCO/cuDSS code.
''')
validation=dict(npm_check_passed=True,selected_dataset='gtoc12-geometry-v659',browser_tab='29',webgl2=True,ships=23,collected_asteroids=195,deployed_asteroids=196,replay_samples=11679,context_points=73562,max_asteroid_kepler_error_km=3.59e-6,max_earth_kepler_error_km=3.07e-7,fleet_sha256=hashlib.sha256((viewer/'fleet.json').read_bytes()).hexdigest(),solution_sha256=hashlib.sha256((root/'h100-best/Result.txt').read_bytes()).hexdigest())
(root/'viewer-validation.json').write_text(json.dumps(validation,indent=2))
manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'sha256.json'}
(root/'sha256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(files=len(manifest),bytes=sum(r['bytes'] for r in manifest.values()))))
# Stage a reviewable README patch containing this tranche alone, preserving
# concurrent edits to its other paragraphs in the shared working directory.
base=subprocess.check_output(['git','show','HEAD:README.md'],text=True)
live=Path('README.md').read_text()
start=live.index('An additional opt-in path keeps joint preflight')
end=live.index('\n\n',start)
addition=live[start:end]+'\n\n'
anchor='The richer family32 search reproduces'
assert addition not in base and anchor in base
desired=base.replace(anchor,addition+anchor,1)
patch=''.join(difflib.unified_diff(base.splitlines(True),desired.splitlines(True),fromfile='a/README.md',tofile='b/README.md'))
Path('build/performance/readme-geometry-v659.patch').write_text(patch)
