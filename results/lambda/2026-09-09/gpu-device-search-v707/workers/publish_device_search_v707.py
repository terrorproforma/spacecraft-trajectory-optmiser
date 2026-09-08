from pathlib import Path
import hashlib
import json
import shutil
import tarfile

out=Path('results/lambda/2026-09-09/gpu-device-search-v707');out.mkdir(parents=True,exist_ok=False)
archives={'local':Path('/home/angus/device-search-v707-local.tar.gz'),'h100':Path('build/performance/retrieved-device-search-v707/device-search-v707-h100.tar.gz')}
opened={}
for machine,path in archives.items():
    record=json.loads(path.with_suffix('.record.json').read_text());manifest=json.loads(path.with_suffix('.manifest.json').read_text())
    raw=path.read_bytes();assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
    shutil.copyfile(path,out/(machine+'-raw.tar.gz'))
    tar=tarfile.open(path);opened[machine]=tar
    seen=set()
    for member in tar:
        assert member.isfile() and member.name in manifest and member.name not in seen
        seen.add(member.name);blob=tar.extractfile(member).read();m=manifest[member.name]
        assert len(blob)==m['bytes'] and hashlib.sha256(blob).hexdigest()==m['sha256'],member.name
    assert seen==set(manifest)
    (out/(machine+'-raw.manifest.json')).write_text(json.dumps(manifest,indent=2))
    (out/(machine+'-raw.record.json')).write_text(json.dumps(record,indent=2))
def raw(machine,name):return opened[machine].extractfile(name).read()
def read(machine,name):return json.loads(raw(machine,name))
def extract(machine,name,target):
    p=out/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw(machine,name))
for machine in archives:
    for name in ['report.json','pytest.log','memcheck.log','worker.py','source-manifest.json']:
        extract(machine,'final-controller/'+name,machine+'/final/'+name)
    extra='followup-v704' if machine=='local' else 'followup-v705'
    for name in ['report.json','benchmark.json','benchmark.log']:
        extract(machine,'final-controller/'+extra+'/'+name,machine+'/followup/'+name)
    for mode in ['racecheck','synccheck']:
        prefix='final-controller/followup-v704/' if machine=='local' else 'final-controller/'
        extract(machine,prefix+mode+'.log',machine+'/final/'+mode+'.log')
    extract(machine,'campaign/report.json',machine+'/campaign.json')
    extract(machine,'campaign/launch-v706.json',machine+'/launch.json')
for name in ['Result.txt','viewer/manifest.json','viewer/trajectories.json']:
    extract('h100','campaign/candidate0/best/'+name,'h100-best/'+name)
extract('h100','campaign/candidate0/report.json','h100-best/campaign-report.json')
sources={}
manifest=read('local','final-controller/source-manifest.json')
remote=read('h100','final-controller/source-manifest.json')
for name in ['cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h','cpp/cuda/src/gtoc12_joint.cu','src/spacepdhcg/gtoc12/gpu_joint.py','src/spacepdhcg/gtoc12/jointopt.py','tests/test_gtoc12_gpu_joint_search.py']:
    digest=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert digest==manifest['repo/'+name]==remote['repo/'+name],name
    sources[name]=digest
(out/'published-source.json').write_text(json.dumps(dict(files=sources,base='3175fb4497922bb4ada2f236c8f0d562fc22a4e4',note='Final source identical on both GPUs. Concurrent persistent-backend changes excluded from frozen builds.'),indent=2))
summary=dict(source=sources,benchmarks={},campaigns={},libraries={},limitations=['Local active-loop memcheck returns CUDA error 999; active-loop race/sync and empty-loop memcheck pass. H100 active-loop memory/race/sync checks pass.'],default_enabled=False)
for machine in archives:
    extra='followup-v704' if machine=='local' else 'followup-v705'
    benchmark=read(machine,'final-controller/'+extra+'/benchmark.json');assert benchmark['complete']
    summary['benchmarks'][machine]=benchmark['comparisons']
    campaign=read(machine,'campaign/report.json');assert campaign['complete'] and campaign['success']
    summary['campaigns'][machine]=[]
    for run in campaign['runs']:
        assert run['best']['official']['ok'] and run['best']['independent']['ok'] and not run['nonconverged']
        summary['campaigns'][machine].append({k:v for k,v in run.items() if k!='command'})
    summary['libraries'][machine]=dict(core=campaign['core_sha256'],qoco=campaign['qoco_sha256'])
(out/'summary.json').write_text(json.dumps(summary,indent=2))
(out/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n')
workers=out/'workers';workers.mkdir()
for p in Path('build/performance').glob('*'):
    if p.is_file() and any(term in p.name for term in ['joint_search_v697','joint_search_v698','joint_search_v700','joint_search_v701','joint_search_v702','joint_search_v704','joint_search_v703','joint_search_v705','search_campaign_v703','search_campaign_v706','device_search_v707','search_h100_v701','search_h100_v705']):
        if p.suffix in ('.py','.mjs'):shutil.copyfile(p,workers/p.name)
for tar in opened.values():tar.close()
print(json.dumps(dict(output=str(out),benchmarks=summary['benchmarks'],campaigns={m:[dict(name=r['name'],seconds=r['process_seconds'],score=r['best']['score_kg']) for r in rows] for m,rows in summary['campaigns'].items()})),flush=True)
