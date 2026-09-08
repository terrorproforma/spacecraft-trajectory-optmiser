"""Retain both bounded experiments, including observation failures and all outputs."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

live = Path(__file__).resolve().parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
definitions = [('automatic-cold-v610', 'auto-cold-v610'), ('upstream-zero-quadratic-v613', 'upstream-zero-q-v613')]
for destination_name, source_name in definitions:
    source = live / 'build/performance' / source_name
    destination = live / 'results/local/2026-09-09' / destination_name
    destination.mkdir(exist_ok=True)
    if (destination / 'sha256.json').exists():
        previous = json.loads((destination / 'sha256.json').read_text())
        for name, info in previous.items():
            assert sha(destination / name) == info['sha256'], name
    report = json.loads((source / 'run/report.json').read_text())
    assert report['complete'] and len(report['cases']) == 4
    source_index = {}
    members = {}
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name == 'upstream_snapshot_replay' or '__pycache__' in path.parts:
            continue
        rel = path.relative_to(source).as_posix()
        source_index[rel] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        if rel.startswith('run/') and path.suffix == '.log':
            members[rel] = source_index[rel]
        else:
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    with tarfile.open(destination / 'raw-logs.tar.gz', 'w:gz') as archive:
        for rel in members:
            archive.add(source / rel, arcname=rel)
    with tarfile.open(destination / 'raw-logs.tar.gz', 'r:gz') as archive:
        assert set(archive.getnames()) == set(members)
        for rel, info in members.items():
            data = archive.extractfile(rel).read()
            assert len(data) == info['bytes'] and hashlib.sha256(data).hexdigest() == info['sha256']
    (destination / 'raw-log-members.json').write_text(json.dumps(members, indent=2) + '\n')
    (destination / 'source-files.json').write_text(json.dumps(source_index, indent=2) + '\n')
    if source_name == 'auto-cold-v610':
        audit_dir = live / 'build/performance/cold-convergence-audit-v610'
        (destination / 'analysis').mkdir(exist_ok=True)
        for name in ('analyze_scaling.py', 'scaling-findings.json', 'analyze_saved_cold.py', 'saved-cold-findings.json', 'UPSTREAM_REVIEW.md'):
            shutil.copyfile(audit_dir / name, destination / 'analysis' / name)
        readme = '''# Automatic-grid cold comparison v610

Four actual cold calls use the unchanged v609d core and v609e executable, with
native automatic grid selection, 100,000 iterations and a30-second deadline each.
All finish at the iteration cap and fail the unchanged original-equation gate.
The optional common-KKT policy follows the same numerical trajectory to FP64
rounding differences. This is an acceptance-overhead diagnostic, not a qualified
throughput result, cold-convergence improvement, mission score or SOTA claim.

| Capture | Policy | Iterations | Native solve seconds | Original normalized gap |
|---|---|---:|---:|---:|
'''
        for row in report['cases']:
            readme += f"| {row['name']} | {'common KKT' if row['common'] else 'natural'} | {row['final']['iterations']} | {row['final']['solve_seconds']:.9f} | {row['audit']['gap']:.12g} |\n"
        readme += '''
No supplied-point bootstraps, retries or recovery calls occurred. The first solve
completed before the original results-parser rejected a null grid-count field.
The resumed runner reuses that exact hashed output and executes only the remaining
three calls. The executable records a null requested grid when automatic selection
is used; it does not export the effective grid count. The original failure report,
both runner versions and all four raw outputs are retained.

The separate CPU analysis re-evaluates prior archived cold vectors and the frozen
scaling implementation. Numerical spectral estimates put the two zero-Q stability
products below one; this is not a formal general spectral certificate. Its gap
decomposition identifies substantive equality/stationarity/complementarity errors,
and an exact L1 epigraph-prox reformulation hypothesis for future work.

The native binaries and source lineage are identified in run/report.json and the
[v609 evidence](../persistent-common-kkt-v609/README.md). Physics and score remain
at the incumbent; these conic captures do not certify nonlinear mission dynamics.
'''
    else:
        readme = '''# Exact-zero quadratic reference dispatch v613

The opt-in `--omit-zero-quadratic` diagnostic passes a null quadratic descriptor
only after verifying every canonical Hessian coefficient equals exactly zero.
Original snapshot bytes, objective, constraints, cone mapping and independent KKT
thresholds remain unchanged. The default comparator still retains the original
full symmetric CSC descriptor. Nonzero quadratic fixtures and duplicate flags
are rejected with CUDA hidden; all five CPU validation checks pass.

Four native calls use the pinned upstream commit167c8b72b4b96d2f94d405b8763e485514192b81:
two supplied qualified points accepted at zero iterations, followed by two cold
100,000-iteration/60-second runs. Both cold runs remain unqualified.

| Capture | Iterations | Reported inner counter | Native C API wall seconds | Original normalized gap | Qualified |
|---|---:|---:|---:|---:|---|
'''
        for row in report['cases']:
            readme += f"| {row['name']} | {row['final']['iterations']} | {row['final']['inner_iterations']} | {row['final']['solve_wall_seconds']:.9f} | {row['audit']['gap']:.12g} | {row['audit']['qualified']} |\n"
        readme += '''
The saved [v608 calls](../upstream-identical-capture-v608/README.md) used the same
snapshots, upstream commit, accuracy settings and cold iteration/time caps, with
the structurally nonempty zero Hessian. Their conditioning/difficult C API times
were37.332045778/34.817582406seconds, versus10.315961813/8.796659007 here. These are
single diagnostic samples including native setup, host transfers and synchronization;
neither path produced a qualified cold result, so this is not a qualified solver
speedup or a mission improvement.

`inner_iterations` faithfully exports upstream's counter. Its source increments
`inner_solver->total_count` once unconditionally per outer update, even on NON_Q.
Thus100,000 is expected here despite no BB inner solve. The earlier general-Q
path reported200,000, including the unconditional counts. The initial harness
expected zero, failed after the third completed call, and was corrected. All
three completed outputs were reused; only the fourth outstanding call was then
launched. Both harnesses, the observation failure and every raw output are retained.

The mathematical dispatch proof is in pinned upstream `src/utils.cu:834`,
`src/solver_state.cu:307` and `src/pdhg_core_op.cu:727`. No upstream source was
changed. The static reference library identity and build command are recorded in
manifest.json; no executable or library is included in this evidence package.
'''
    (destination / 'README.md').write_text(readme.replace('a30-', 'a 30-').replace('commit167', 'commit 167').replace('were37', 'were 37').replace('/34.', ' / 34.').replace('seconds,', ' seconds,').replace('versus10.', 'versus 10.').replace('/8.', ' / 8.').replace('Thus100,000', 'Thus 100,000').replace('reported200,000', 'reported 200,000'))
    (destination / '.gitattributes').write_text('* -text\n')
    shutil.copyfile(__file__, destination / 'package.py')
    index = {p.relative_to(destination).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(destination.rglob('*')) if p.is_file() and p != destination / 'sha256.json'}
    assert not any(Path(name).suffix in ('.so', '.exe', '.dll', '.o', '.pyc') for name in index)
    (destination / 'sha256.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'artifact': str(destination), 'indexed_files': len(index), 'bytes': sum(p['bytes'] for p in index.values()), 'index_sha256': sha(destination / 'sha256.json')}))
