from pathlib import Path
import hashlib
import json
import shutil
import tarfile

p = Path('build/performance')
base = Path('results/lambda/2026-09-08')
target = base / 'gpu-conditioning-local-v528'

def read(folder, name='report.json'):
    return json.loads((p / folder / name).read_text())

local = ['conditioning-v519', 'conditioning-legs-v521', 'conditioning-campaign-v523',
         'conditioning-lost-v525', 'conditioning-qp-v526',
         'conditioning-qp-reg-v527', 'conditioning-qp-ir-v528']
remote = ['conditioning-v520', 'conditioning-legs-v522', 'conditioning-campaign-v524']
for name in local + ['retrieved-' + tag for tag in remote]:
    r = read(name)
    assert r['complete'] and not r.get('error'), name
for folder, count in [('conditioning-legs-v521', 191), ('retrieved-conditioning-legs-v522', 192)]:
    r = read(folder, 'analysis.json')
    assert r['baseline']['converged'] == 205 and r['candidate']['converged'] == count
    assert len(r['lost_baseline']) == 205 - count
for folder in ['conditioning-campaign-v523', 'retrieved-conditioning-campaign-v524']:
    rows = read(folder, 'analysis.json')['rows']
    assert len(rows) == 4
    assert all(r['same_initial_plans'] and r['same_logical_counts'] for r in rows)
    assert all(abs(r['score'] - 548.2546201232) < 1e-6 for r in rows)
assert read('conditioning-qp-reg-v527')['same_numerical_qp']
assert all(not a['qualified'] for r in read('conditioning-qp-ir-v528')['cases'] for a in r['audits'])

target.mkdir(exist_ok=False)
archives = {}
for name in local:
    root = p / name
    members = {q.relative_to(root).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest()
               for q in sorted(root.rglob('*')) if q.is_file()}
    archive_name = name + '.tar.gz'
    with tarfile.open(target / archive_name, 'w:gz') as archive:
        for member in members:
            archive.add(root / member, arcname=member, recursive=False)
    archives[archive_name] = members
(target / 'archive-manifests.json').write_text(json.dumps(archives, indent=2))
summary = dict(source_base_commit='3939986797e3ee4c4c59a22d2ef7e539e62642eb',
               production_settings_changed=False,
               local_reports={name: read(name) for name in local},
               analyses={name: read(name, 'analysis.json') for name in
                         ['conditioning-v519', 'conditioning-legs-v521', 'conditioning-campaign-v523'] +
                         ['retrieved-' + tag for tag in remote]},
               limitations=[
                   'Negative qualification results: two Ruiz passes lose previously certified legs and slow full campaigns.',
                   'Existing workspace pool eligibility differs: zero Ruiz reuses storage, scaled workspaces rebuild.',
                   'Pilot budgets are 30 seconds per leg; full replay preserves original fixture budgets.',
                   'QP regularization and iterative refinement sweeps are diagnostic only; no acceptance limits changed.',
                   'H100 common-certified case 171 differs by 0.01735 kg with scaling; no mass-equivalence claim is made.',
                   'No new fleet record, official leaderboard submission, or fully GPU-controlled application is established.',
               ])
(target / 'summary.json').write_text(json.dumps(summary, indent=2))
for pattern in ['*conditioning*.py']:
    for q in p.glob(pattern):
        if 'v133' in q.name:
            continue
        shutil.copy2(q, target / q.name)
for name in ['analyse_qps_v169.py', 'solver_phase_details.py', 'remote_exec.py']:
    shutil.copy2(p / name, target / name)
folders = [target]
for tag in remote:
    root = base / ('gpu-' + tag)
    folders.append(root)
    for name in ['report.json', 'analysis.json']:
        shutil.copy2(p / ('retrieved-' + tag) / name, root / name)
for root in folders:
    (root / '.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
    manifest = {q.relative_to(root).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest()
                for q in sorted(root.rglob('*')) if q.is_file() and q.name != 'files-sha256.json'}
    assert all((root / name).stat().st_size < 90_000_000 for name in manifest)
    (root / 'files-sha256.json').write_text(json.dumps(manifest, indent=2))
    print(root, len(manifest), 'files')
