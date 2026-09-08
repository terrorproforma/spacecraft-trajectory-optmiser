from pathlib import Path
import hashlib,json,shutil
out=Path('results/lambda/2026-09-09/gpu-leg-certificate-v712')
viewer=Path('results/lambda/2026-09-06/visualiser/data/gtoc12-certificate-v712')
shutil.copytree(viewer,out/'viewer-dataset')
(out/'viewer-validation.json').write_text(json.dumps(dict(
    dataset='gtoc12-certificate-v712',url='http://127.0.0.1:4173/?dataset=gtoc12-certificate-v712&epoch=69807&preset=oblique&z=1',
    selected_label='H100 CUDA certificates v712',ships=23,deployed_asteroids=196,collected_asteroids=195,
    displayed_weighted_kg=12810.136,visible_scene=True,renderer='Active WebGL2 / RTX 5090',
    validation='npm run check passed all installed datasets; CUA accessibility tree and screenshot inspected',
    browser_console_inspected=False,samples=11679,context_points=73562,
    asteroid_crosscheck_max_km=3.59e-6,earth_crosscheck_max_km=3.07e-7),indent=2))
for name in ('prepare_certificate_viewer_v712.mjs','finalize_certificate_v712.py'):
    shutil.copyfile(Path('build/performance')/name,out/'workers'/name)
(out/'README.md').write_text('''# Native CUDA leg certificates and campaign profiles

See [the implementation and results](../../../../docs/GPU_LEG_CERTIFICATION.md).
The whole application is not yet GPU controlled. Native CUDA refinement now
defaults to CUDA DOP853 leg certification; CPU is an explicit ablation. All
published fleet outputs retain fresh independent CPU and official checks.

The summary records four complete campaigns, sequential certificate benchmarks,
actual route certificate backends, native library hashes and published source
hashes. All 36 native solves converge in each campaign. Score remains
12,810.135953 weighted kg; there is no meaningful score gain.

Each raw archive has a hash/length record and a complete member manifest. Every
member was independently read and verified after retrieval. Archives preserve
v708 profiles and their outputs, v709 frozen Python/test sources and full paired
campaigns, final CLI follow-up sources, benchmarks, successful checks and the
original failed follow-up launcher/traceback. The launcher failure occurred before
tests or GPU work; v711 fixes its source-prefix extraction. The successful
follow-up report does not rewrite the original worker or log.

Native libraries are the unchanged v702 binaries already archived in
`../gpu-device-search-v707/`: RTX
`65f1335e3af820d22451e0d4f587f2d26aeccf364a0ed3bb9597bfd009bac5aa`, H100
`4288e12e3cf77664ca4e305778a11b4d13662f30474585844db0fb521f2d60a3`.
The prepared QOCO binaries and frozen campaign fixture files are also retained in
that prior evidence. Their existing paths are referenced by the original workers;
the archived workers preserve the executed context, not a portable one-command
installer. No modified persistent solver was loaded in these runs.

The v709 source is the tested bridge/driver. Only CLI preflight/reporting and its
test change in followup-v710; those final files pass the recorded v711 tests.
The source hashes in summary.json identify the exact final published files.

Benchmark scope: two real routes, 19 emitted event-to-event legs each, sequential
certificate calls with retained CUDA buffers, five alternating pairs (first is
warmup). Includes packing and result construction; excludes solving, parsing,
original sample emission and full mission bookkeeping. Final numeric certificates
and every timing sample are saved. Full campaigns have one pair per GPU, so small
end-to-end differences do not establish speedup. cProfile timings include its
overhead and are diagnostic only.

The H100 best result is `h100-best/Result.txt`. `h100-best/viewer/` is its
independently propagated export; `viewer-dataset/` is the exact imported local
viewer dataset. The live label is **H100 CUDA certificates v712** at
`http://127.0.0.1:4173/?dataset=gtoc12-certificate-v712&epoch=69807&preset=oblique&z=1`.
''')
manifest={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='sha256.json'}
(out/'sha256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(files=len(manifest),bytes=sum(r['bytes'] for r in manifest.values()))))
