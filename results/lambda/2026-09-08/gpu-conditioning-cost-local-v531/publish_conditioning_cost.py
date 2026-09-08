from pathlib import Path
import hashlib,json,shutil,tarfile
p=Path('build/performance');base=Path('results/lambda/2026-09-08')
target=base/'gpu-conditioning-cost-local-v531'
remote=base/'gpu-conditioning-paths-v532'
retrieved=p/'retrieved-conditioning-paths-v532'
tags=['conditioning-paths-v529','conditioning-matrices-v530','conditioning-cost-v531']
def read(path):return json.loads(path.read_text())
def validate(root):
 reports={tag:read(root/tag/'report.json') for tag in tags}
 for r in reports.values():assert r['complete'] and not r.get('error')
 for row in reports[tags[0]]['cases']:
  assert len(row['audits'])==3
  assert sum(a['qualified'] for a in row['audits'])==(3 if row['ruiz']==0 else 0)
 for row in reports[tags[1]]['cases']:
  assert max(row['scaled_data_relative_errors'].values())<1e-14
  assert row['audit']['qualified']==(row['ruiz']==0)
 assert reports[tags[1]]['clarabel']['audit']['qualified']
 for row in reports[tags[2]]['cases']:
  assert len(row['audits'])==3
  assert sum(a['qualified'] for a in row['audits'])==(3 if row['cost_scale']==1 else 0)
 return reports
local_reports=validate(p);remote_reports=validate(retrieved/'build/performance')
assert read(retrieved/'report.json')['complete']
target.mkdir(exist_ok=False)
archives={}
for tag in tags:
 root=p/tag
 members={q.relative_to(root).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(root.rglob('*')) if q.is_file()}
 archive_name=tag+'.tar.gz'
 with tarfile.open(target/archive_name,'w:gz') as t:
  for name in members:t.add(root/name,arcname=name,recursive=False)
 archives[archive_name]=members
(target/'archive-manifests.json').write_text(json.dumps(archives,indent=2))
scripts=['replay_conditioning_paths_v529.py','inspect_conditioning_v530.py','isolate_conditioning_cost_v531.py','analyse_qps_v169.py',
         'prepare_conditioning_paths_v532.py','run_conditioning_paths_v532.py','launch_conditioning_paths_v532.py',
         'archive_conditioning_paths_v532.py','retrieve_conditioning_paths_v532.py','publish_conditioning_cost.py','verify_conditioning_cost_blobs.py']
for name in scripts:shutil.copy2(p/name,target/name)
for root,reports in [(target,local_reports),(remote,remote_reports)]:
 summary=dict(source_base_commit='66c039a158bc234a80b106774bc426ba10a5b1a3',production_settings_changed=False,reports=reports,
              finding='Objective multiplier 0.0001 alone reproduces the captured failure; variable and constraint scaling without that multiplier qualifies all sampled trials.',
              limitations=['Single captured QP; not complete-workload performance or a general solver fix.','Direct setup and instrumented inspections use CPU work; these are diagnostic comparisons only.',
                           'Original-unit independent accuracy gates unchanged; externally scaled runs tighten native tolerances with the objective multiplier.',
                           'No new fleet score; no production numerical setting promoted.'])
 (root/'summary.json').write_text(json.dumps(summary,indent=2))
 if root==remote:shutil.copy2(retrieved/'report.json',root/'report.json')
 (root/'.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
 manifest={q.relative_to(root).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(root.rglob('*')) if q.is_file() and q.name!='files-sha256.json'}
 assert all((root/name).stat().st_size<90_000_000 for name in manifest)
 (root/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
 print(root,len(manifest),'files')
