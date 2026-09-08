from pathlib import Path
import hashlib
import json
import shutil
import statistics
import tarfile

out=Path('results/lambda/2026-09-09/gpu-conditioning-retry-v696');out.mkdir(parents=True,exist_ok=False)
archives={'local':Path('/home/angus/retry-conditioning-v696-local.tar.gz'),'h100':Path('build/performance/retrieved-retry-v696/retry-conditioning-v696-h100.tar.gz')}
opened={}
for name,p in archives.items():
    target=out/(name+'-raw.tar.gz');shutil.copyfile(p,target)
    manifest=json.loads(p.with_suffix('.manifest.json').read_text());record=json.loads(p.with_suffix('.record.json').read_text())
    assert target.stat().st_size==record['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==record['sha256']
    tar=tarfile.open(target);opened[name]=tar
    for member in tar:
        assert member.isfile() and member.name in manifest
        raw=tar.extractfile(member).read();r=manifest[member.name]
        assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'],member.name
    (out/(name+'-raw.manifest.json')).write_text(json.dumps(manifest,indent=2));(out/(name+'-raw.record.json')).write_text(json.dumps(record,indent=2))
def raw(machine,name):return opened[machine].extractfile(name).read()
def read(machine,name):return json.loads(raw(machine,name))
def extract(machine,name,target):
    path=out/target;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw(machine,name))
local=['build/report.json','build/validation-v684/report.json','build/validation-v687/report.json','build/validation-v687/new.log','build/validation-v687/regression.log','reporting/report.json','reporting/pytest.log','screening/report.json','fixed-normalization/output/report.json','return-replay/output/report.json','leg-replay/comparison.json','leg-replay/report.json','build/campaign-v694/report.json','build/campaign-v691/report.json','build/campaign-v691/baseline0.log']
h100=['report.json','return-output/report.json','broader-v689/report.json','broader-v689/comparison.json','broader-v689/regression.log','reporting-v692/report.json','reporting-v692/pytest.log','reporting-v692/memcheck.log','campaign-v695/report.json','campaign-v693/report.json']
for machine,names in [('local',local),('h100',h100)]:
    for name in names:extract(machine,name,machine+'/'+name)
for name in ['Result.txt','viewer/manifest.json','viewer/trajectories.json']:
    extract('h100','campaign-v695/candidate0/best/'+name,'h100-best/'+name)
extract('h100','campaign-v695/candidate0/report.json','h100-best/campaign-report.json')
campaigns=[]
for machine,prefix,labels in [('local','build/campaign-v694',['baseline0','candidate0','candidate1','baseline1']),('h100','campaign-v695',['candidate0'])]:
    root=read(machine,prefix+'/report.json');assert root['complete'] and root['success']
    for label in labels:
        run=read(machine,prefix+'/'+label+'/report.json');assert run['complete'] and run['best']['official']['ok'] and run['best']['independent']['ok']
        native=[json.loads(line) for line in raw(machine,prefix+'/'+label+'/native-solves.jsonl').decode().splitlines()]
        timed=next(r for r in root['runs'] if r['name']==label)
        campaigns.append(dict(machine=machine,name=label,process_seconds=timed['process_seconds'],score_kg=run['best']['score_kg'],raw_kg=run['best']['total_mass_kg'],orders=run['orders'],native_solves=len(native),native_nonconverged=[dict(solve=r['solve'],status=r['status'],iterations=r['iterations']) for r in native if r['status']!='converged'],native_seconds=sum(r['seconds'] for r in native),native_iterations=sum(r['iterations'] for r in native),independent=run['best']['independent'],official=run['best']['official']))
source={}
compiled=read('local','build/report.json')['source_sha256']
for name,digest in compiled.items():
    if name.startswith('scripts/'):continue
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    source[name]=digest
for name,digest in read('local','reporting/report.json')['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    source[name]=digest
(out/'published-source.json').write_text(json.dumps(dict(files=source,native_base='e7da7d9e7a25aa4a8309f5c80e516f0d4852dd8a',note='Prepared vendor source reproduced byte-for-byte by the final formatted preparation tool; final Python reporting changes verified separately with unchanged native binaries.'),indent=2))
summary=dict(source=source,archives={name:read(name,'build/report.json' if name=='local' else 'report.json')['libraries'] for name in archives},leg_replay={name:read(name,'leg-replay/comparison.json' if name=='local' else 'broader-v689/comparison.json') for name in archives},campaigns=campaigns,local_process_medians={kind:statistics.median(r['process_seconds'] for r in campaigns if r['machine']=='local' and r['name'].startswith(kind)) for kind in ('baseline','candidate')},physics_tolerances_changed=False,extra_solver_attempts_added=False,default_enabled=False)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
workers=out/'workers';workers.mkdir()
for name in ['screen_return_conditioning_v681.py','return_normalized_v682.py','build_retry_conditioning_v683.py','check_retry_conditioning_v684.py','return_conditioning_retry_v685.py','upload_retry_h100_v686.py','validate_retry_conditioning_v687.py','replay_retry_fleet_legs_v688.py','followup_retry_h100_v689.py','validate_retry_reports_v690.py','prepare_retry_campaign_v691.py','upload_retry_reporting_v692.py','fix_retry_campaign_fixtures_v694.py','restart_retry_h100_campaign_v695.py','compare_retry_legs_v688.py','archive_retry_h100_v696.py','archive_retry_local_v696.py','collect_retry_h100_v696.py','publish_retry_v696.py']:
    shutil.copyfile(Path('build/performance')/name,workers/name)
(out/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n')
for tar in opened.values():tar.close()
print(json.dumps(dict(output=str(out),leg_replay=summary['leg_replay'],local_process_medians=summary['local_process_medians'],campaigns=[{k:r[k] for k in ['machine','name','native_nonconverged','score_kg','native_seconds','native_iterations']} for r in campaigns])))
