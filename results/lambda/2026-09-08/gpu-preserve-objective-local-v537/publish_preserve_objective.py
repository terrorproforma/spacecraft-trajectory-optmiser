from pathlib import Path
import hashlib,json,shutil,tarfile
p=Path('build/performance');base=Path('results/lambda/2026-09-08');target=base/'gpu-preserve-objective-local-v537'
def read(path):return json.loads(path.read_text())
local=read(p/'preserve-objective-v536/analysis.json');remote=read(p/'retrieved-preserve-objective-v535/validation/analysis.json')
for r in [local,remote]:
 assert not r['lost_baseline'] and r['baseline']['converged']==r['candidate']['converged']==205
 assert r['max_certified_mass_delta']['delta_kg']<1e-5
for root in [p/'preserve-campaign-v537',p/'retrieved-preserve-campaign-v538']:
 r=read(root/'report.json');assert r['complete'] and not r.get('error')
 assert '107 passed' in (root/'pytest.log').read_text()
 rows=read(root/'analysis.json')['rows'];assert len(rows)==4
 assert all(row['same_initial_plans'] and row['same_logical_counts'] and abs(row['score']-548.2546201232)<1e-6 for row in rows)
for root in [p/'preserve-objective-v536',p/'retrieved-preserve-objective-v535/validation']:
 r=read(root/'report.json');assert r['complete'] and not r.get('error')
 assert all(a['qualified'] for row in r['qp'] if row['enabled']=='1' for a in row['audits'])
 for name in ['memcheck','synccheck']:assert 'ERROR SUMMARY: 0 errors' in (root/(name+'.log')).read_text()
 assert '0 errors, 0 warnings' in (root/'racecheck.log').read_text()
target.mkdir(exist_ok=False);archives={}
for tag in ['preserve-build-v534','preserve-objective-v536','preserve-campaign-v537']:
 root=p/tag;r=read(root/'report.json');assert r['complete'] and not r.get('error')
 members={q.relative_to(root).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(root.rglob('*')) if q.is_file()}
 name=tag+'.tar.gz'
 with tarfile.open(target/name,'w:gz') as t:
  for member in members:t.add(root/member,arcname=member,recursive=False)
 archives[name]=members
(target/'archive-manifests.json').write_text(json.dumps(archives,indent=2))
summary=dict(default_enabled=False,source_base_commit='71fb2b51b798ff0cb8b2b9002f5ed9a056f1de20',flag='SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE',
 local_legs=local,lambda_legs=remote,local_campaign=read(p/'preserve-campaign-v537/analysis.json'),lambda_campaign=read(p/'retrieved-preserve-campaign-v538/analysis.json'),
 final_preparation=read(p/'preserve-preparation-verification.json'),
 limitations=['Opt-in accuracy fix; zero-Ruiz production default retained.','205/225 qualified on both GPUs; all remaining attempts remain rejected.',
              'Scaled workspaces still rebuild; no broad performance improvement established.','No new fleet record or fully GPU-controlled route/fleet application.'])
(target/'summary.json').write_text(json.dumps(summary,indent=2))
for pattern in ['*preserve*objective*v53*.py','*preserve*campaign*v53*.py','analyze_preserve*.py','prepare_preserve_analysis.py',
                'verify_preserve_preparation.py','freeze_preserve*.py','publish_preserve_objective.py','verify_preserve_objective_blobs.py']:
 for q in p.glob(pattern):shutil.copy2(q,target/q.name)
shutil.copy2(p/'preserve-preparation-verification.json',target/'preserve-preparation-verification.json')
folders=[target]
for tag,analysis_path in [('preserve-objective-v535','validation/analysis.json'),('preserve-campaign-v538','analysis.json')]:
 root=base/('gpu-'+tag);folders.append(root)
 for name,path in [('report.json','report.json'),('analysis.json',analysis_path)]:shutil.copy2(p/('retrieved-'+tag)/path,root/name)
for root in folders:
 (root/'.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
 manifest={q.relative_to(root).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(root.rglob('*')) if q.is_file() and q.name!='files-sha256.json'}
 assert all((root/name).stat().st_size<90_000_000 for name in manifest)
 (root/'files-sha256.json').write_text(json.dumps(manifest,indent=2));print(root,len(manifest),'files')
