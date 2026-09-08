from pathlib import Path
import hashlib
import json
import shutil
import tarfile

p = Path('build/performance')
target = Path('results/lambda/2026-09-08/gpu-pipeline-profile-v510')
assert not target.exists()
reports = {}
for name in ['pipeline-profile-v509', 'solver-phase-v510']:
    root = p / name
    r = json.loads((root / 'report.json').read_text())
    assert r['complete'] and not r.get('error') and r['returncode'] == 0
    for source, digest in r['source_sha256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, source
    result = json.loads((root / 'output/run_report.json').read_text())
    assert all(result['best'][k]['ok'] for k in ['official', 'independent'])
    reports[name] = r

root = p / 'solver-phase-v510'
phases = [json.loads(line.split('SCVX_PHASE ', 1)[1])
          for line in (root / 'campaign.log').read_text().splitlines()
          if line.startswith('SCVX_PHASE ')]
calls = json.loads((root / 'calls.json').read_text())
assert len(phases) == len(calls) == 47
for phase, call in zip(phases, calls, strict=True):
    assert phase['intervals'] == call['nodes'] - 1
    assert phase['iterations'] == call['iterations']
summary = dict(
    source_commit=reports['solver-phase-v510']['source_commit'],
    scope='Two diagnostic campaigns, not a matched speed comparison. Native host-wall phases use existing synchronization points; they are not isolated kernel timings.',
    profile_process_seconds=reports['pipeline-profile-v509']['seconds'],
    phase_process_seconds=reports['solver-phase-v510']['seconds'],
    calls=len(calls),
    solver_wall_seconds=sum(c['seconds'] for c in calls),
    phase_seconds={key: sum(row[key] for row in phases) for key in
                   ['setup', 'priming', 'graph_build', 'graph_run', 'graph_close', 'download', 'cleanup']},
    statuses={status: sum(c['status'] == status for c in calls)
              for status in sorted({c['status'] for c in calls})},
    limitations=[
        'The first profile recorded a 4.06-second single-leg outlier. The detailed follow-up did not reproduce it; its slowest leg took 0.877 seconds.',
        'Priming remains the largest native phase and includes initial numerical solves, not just allocation.',
        'One leg remained stationary with nonzero defects; it was rejected. No solver tolerance or acceptance criterion changed.',
        'Both campaigns passed official and independent mission checks. They did not produce a new best fleet.',
    ])
target.mkdir(parents=True)
(target / 'summary.json').write_text(json.dumps(summary, indent=2))
archives = {}
for name in reports:
    root = p / name
    members = {q.relative_to(root).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest()
               for q in sorted(root.rglob('*')) if q.is_file()}
    with tarfile.open(target / (name + '.tar.gz'), 'w:gz') as archive:
        for member in members:
            archive.add(root / member, arcname=member, recursive=False)
    archives[name + '.tar.gz'] = members
(target / 'archive-manifests.json').write_text(json.dumps(archives, indent=2))
for name in ['run_profile_v509.py', 'profile_v509.py', 'run_solver_phase_v510.py',
             'solver_phase_details.py', 'publish_pipeline_profile_v510.py']:
    shutil.copy2(p / name, target / name)
(target / '.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
manifest = {q.relative_to(target).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest()
            for q in sorted(target.rglob('*')) if q.is_file()}
(target / 'files-sha256.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(summary, indent=2))
