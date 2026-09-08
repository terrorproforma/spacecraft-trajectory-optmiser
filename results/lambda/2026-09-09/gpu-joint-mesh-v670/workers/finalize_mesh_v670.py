from pathlib import Path
import hashlib,json,shutil
root=Path('results/lambda/2026-09-09/gpu-joint-mesh-v670');frozen=Path('/home/angus/spacepdhcg-joint-mesh-v662/repo')
names=['cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h','cpp/cuda/src/gtoc12_joint.cu','src/spacepdhcg/gtoc12/gpu_joint.py','src/spacepdhcg/gtoc12/jointopt.py','tests/test_gtoc12_gpu_joint_mesh.py']
records={}
for name in names:
 data=Path(name).read_bytes();assert data==(frozen/name).read_bytes(),name
 records[name]=hashlib.sha256(data).hexdigest()
(root/'published-source.json').write_text(json.dumps(dict(byte_identical_to_final_tested_source=records,note='The original failed test mistakenly included an extra incumbent row; test-fix-v665.json in the local archive records the correction. Native/runtime source unchanged after build.'),indent=2))
viewer=Path('results/lambda/2026-09-06/visualiser/data/gtoc12-mesh-v670')
shutil.copytree(viewer,root/'viewer-dataset')
for name in ('prepare_mesh_viewer_v670.mjs','finalize_mesh_v670.py','prepare_mesh_publication_v669.py','inspect_mesh_campaign_v669.py'):
 shutil.copy2(Path('build/performance')/name,root/'workers'/name)
(root/'viewer-validation.json').write_text(json.dumps(dict(npm_check_passed=True,browser_tab='30',selected_dataset='gtoc12-mesh-v670',webgl2=True,replay_samples=11679,context_points=73562,ships=23,collected_asteroids=195,deployed_asteroids=196,fleet_sha256=hashlib.sha256((viewer/'fleet.json').read_bytes()).hexdigest(),solution_sha256=hashlib.sha256((root/'h100-best/Result.txt').read_bytes()).hexdigest()),indent=2))
(root/'README.md').write_text('''# CUDA mesh generation and H100 replay

[Implementation, results and loading instructions](../../../../docs/GPU_JOINT_MESH.md).

This opt-in path generates ordered timing moves on CUDA and feeds device epochs
directly into resident geometry and winner selection. Both GPUs pass 87 tests;
25 geometry/mesh tests pass memcheck, synccheck and racecheck on each GPU.
The component gains 7–15% locally and 34–42% on H100 against CPU-generated epochs
with the same resident geometry. These are not whole-mission speedups.

`summary.json` preserves all eight full-fleet process outcomes and timing samples.
Every retained best fleet passes independent and official checks at unchanged
tolerances: 12,810.135953 weighted kg, 14,051.854894 raw kg, 23 ships, 195 collected
asteroids. One local GPU-mesh run and one H100 CPU-mesh run fail to refine the
second, lower-scoring alternative. They retain the verified first result and
skip one full-fleet check, confounding process comparisons. No overall speedup
or score improvement is claimed. All failures remain in the raw archives.

`h100-best/Result.txt` is the downloaded H100 GPU-mesh fleet. Its viewer export,
campaign report and `viewer-dataset` are included. The displayed dataset is:
http://127.0.0.1:4173/?dataset=gtoc12-mesh-v670&epoch=69807&preset=oblique&z=1

The two raw archives contain the frozen source, core/QOCO binaries, fixtures,
build commands, complete test/sanitizer logs, all timing samples and all fleet
attempts. Member and archive hashes verify downloaded bytes. The local archive
also preserves the initial failed test fixture and its corrected final version;
runtime source did not change for that correction. `published-source.json`
verifies the final source against the frozen tested files.

Python still drives mesh progression, deadlines and metadata preparation.
Independent CPU checks gate fleet acceptance. Native conic convergence remains
unreliable on one recorded return: [nine repeated calls and rejected settings](../../../local/2026-09-09/return-replay-v672/).
''')
manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'sha256.json'}
(root/'sha256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(files=len(manifest),bytes=sum(r['bytes'] for r in manifest.values()))))
