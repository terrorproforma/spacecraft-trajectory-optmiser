"""Publish local diagnostic evidence, without binaries, remote writes or score changes."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

repo = Path(__file__).resolve().parents[2]
source = repo / 'build/performance/upstream-snapshot-v608'
dest = repo / 'results/local/2026-09-09/upstream-identical-capture-v608'
dest.mkdir(parents=True, exist_ok=False)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((source / 'manifest.json').read_text())
report = json.loads((source / 'run/report.json').read_text())
assert manifest['complete'] and report['complete'] and report['gpu_calls'] == 8
assert report['qualified_calls'] == 4
assert report['manifest_sha256'] == sha(source / 'manifest.json')
assert manifest['executable_sha256'] == report['executable_sha256'] == sha(source / 'upstream_snapshot_replay')
for name, expected in manifest['source_files_sha256'].items():
    assert sha(source / 'source' / name) == expected, name
for name, expected in manifest['fixtures_sha256'].items():
    assert sha(source / 'fixtures' / name) == expected, name
for name in ['manifest.json', 'build.py', 'build.log']:
    shutil.copyfile(source / name, dest / name)
shutil.copytree(source / 'fixtures', dest / 'fixtures')
shutil.copytree(source / 'run', dest / 'run')
shutil.copyfile(__file__, dest / 'package.py')
with zipfile.ZipFile(dest / 'diagnostic-sources.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(manifest['source_files_sha256']):
        archive.write(source / 'source' / name, name)
table = []
for case in report['cases']:
    r, a = case['result'], case['independent_audit']
    table.append(f"| {case['name']} | {r['iterations']:,} | {r['solve_wall_seconds']:.6f} | {a['primal']:.6g} | {a['dual']:.6g} | {a['gap']:.6g} | {'yes' if a['qualified'] else 'no'} |")
text = """# Pinned upstream identical-capture comparison — v608

The original pinned PDHCG implementation accepts both real known-qualified
GTOC12 conic points at iteration zero. Neither real cold-start problem qualifies
within 100,000 iterations. This separates the known-point stopping defect in
our persistent adapter from a broader cold-convergence problem; it does not
establish a performance win, trajectory certificate or fleet improvement.

Eight native calls ran once on the local RTX 5090 under the shared GPU lock.
Four exact known-point calls (two analytic and two real) pass the unchanged
independent KKT gate. Two analytic cold calls report native OPTIMAL but narrowly
fail the stricter common gap gate. Both real cold calls report ITERATION_LIMIT.
All outcomes, including failures, are retained.

| Case | Iterations | Native API wall seconds | Common primal | Common dual | Common gap | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
""" + '\n'.join(table) + """

## What was held fixed

- Upstream commit `167c8b72b4b96d2f94d405b8763e485514192b81`, tree
  `62b05e6c1bedd385f6c267af3645ae4aae0421b4`, clean source verified locally.
  The existing `0001-free-quadratic-state.patch` is included and hashed.
- The exact original-coordinate captures and QOCO known points from v606.
  Full symmetric Q is constructed once, all scalar inequalities retain their
  dual variables, standard SOC permutations are unchanged, and every upstream
  Pi is negated before the established original-coordinate conversion.
- Native tolerances 1e-9, presolve disabled, upstream default scaling, restart,
  infinity norm and inner iterations retained. Real cold limits are 100,000
  iterations and 60 seconds; real seeded limits are one iteration and 10 seconds.
  A process timeout adds at most 20 seconds to each native time limit.
- Independent original-equation primal/dual/gap and per-block complementarity
  tolerance 1e-9, cone violation 1e-8. The unchanged Python auditor receives only
  a documented termination-code adapter (upstream OPTIMAL is code 1); its
  mathematical formulas and tolerances are not modified. Every C++ qualification
  verdict agrees with the independent Python verdict.

Four hidden-device CPU input checks and one duplicate-option rejection passed.
Analytic cold points verify the dual sign/permutation within 1e-6; that mapping
check does not relax their strict qualification result in the table.

## Interpretation and limits

The older persistent v606 implementation imported the same two real known points
correctly, rejected them under its absolute natural-residual rule and moved away
from them. Upstream's iteration-zero acceptance supports the planned opt-in
common-KKT stopping intervention. Acceptance of a supplied optimum is a
correctness regression test, not evidence of fast optimisation from a cold start.

At 100,000 iterations, upstream's conditioning capture has a much smaller gap
than the old persistent replay; the difficult capture does not improve its gap.
Neither qualifies. Comparing their unqualified wall times as a speedup would be
misleading. The zero-Hessian captures also use explicit linear primal steps;
missing quadratic CG alone cannot explain their failures.

Timing is the synchronous one-shot native C API including its setup, GPU work
and host transfers. It is not pure GPU event time, a warmed repeated-workspace
benchmark, solutions/second, or whole-SCvx time. The recorded 200,000 inner
iterations in each real cold case are upstream telemetry, not 200,000 certified
trajectory solutions. The test does not connect upstream or persistent PDHCG to
the native GTOC12 outer loop. Current GTOC12 refinement still uses GPU QOCO.

The verified 23-ship incumbent remains 14,051.854893908598 raw kg and
12,810.135953048577 fixed-bonus weighted kg. No viewer dataset is promoted.

## Evidence and reproduction

`manifest.json` records the static reference archive, exact compiled source
hashes, compiler/link command and executable hash. `diagnostic-sources.zip`
contains the exact replay/importer, headers, patch, lock and independent auditor;
the upstream tree is identified by the repository lock. Public-header additions
in this source snapshot do not invoke the new persistent policy in the reference
executable. Binaries are retained locally in the build directory, not included
here. `run/report.json`, each raw stdout log and `run/run.py` retain parameters,
initial-point hashes, original vectors and independent results.

The new repository target is `upstream_snapshot_replay` with BUILD_TESTING on.
Run the analytic fixtures before either real capture. Each call requires an
explicit `--iterations` and `--deadline-seconds`; `--initial-point` selects a
separate known-point diagnostic, while `--validate-only` makes no CUDA API call.
Use the shared GPU lock, preserve complete stdout/stderr, and apply the common
external gate to every result. The recorded comparison is not rerun by packaging.
"""
(dest / 'README.md').write_text(text, encoding='utf-8')
index = {str(p.relative_to(dest)).replace('\\', '/'): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
(dest / 'sha256.json').write_text(json.dumps(index, indent=2) + '\n')
print(json.dumps({'artifact': str(dest), 'files': len(index), 'bytes': sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()), 'index_sha256': sha(dest / 'sha256.json')}))
