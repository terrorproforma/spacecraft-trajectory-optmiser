from pathlib import Path
import hashlib,json,shutil,tarfile
out=Path('results/lambda/2026-09-09/gpu-leg-certificate-v712');out.mkdir(parents=True,exist_ok=False)
archives={'local':Path('/home/angus/certificate-v712-local.tar.gz'),'h100':Path('build/performance/retrieved-certificate-v712/certificate-v712-h100.tar.gz')}
opened={};summary=dict(profiles={},campaigns={},benchmarks={},libraries={},sources={})
for machine,path in archives.items():
    record=json.loads(path.with_suffix('.record.json').read_text());manifest=json.loads(path.with_suffix('.manifest.json').read_text())
    assert path.stat().st_size==record['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    tar=tarfile.open(path);opened[machine]=tar;seen=set()
    for member in tar:
        assert member.isfile() and member.name in manifest and member.name not in seen
        seen.add(member.name);raw=tar.extractfile(member).read();row=manifest[member.name]
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],member.name
    assert seen==set(manifest)
    for suffix in ('.tar.gz','.tar.manifest.json','.tar.record.json'):
        original=path if suffix=='.tar.gz' else path.with_suffix('.manifest.json' if 'manifest' in suffix else '.record.json')
        shutil.copyfile(original,out/(machine+'-raw'+suffix))
def raw(machine,name):return opened[machine].extractfile(name).read()
def read(machine,name):return json.loads(raw(machine,name))
def extract(machine,name,target):
    path=out/target;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw(machine,name))
for machine in archives:
    report=read(machine,'certificate/report.json');assert report['complete'] and report['success']
    follow=read(machine,'certificate/followup-v710/report.json');assert follow['complete'] and follow['success']
    summary['profiles'][machine]=read(machine,'profile/profile.json')
    summary['libraries'][machine]=report['core_sha256']
    summary['benchmarks'][machine]=read(machine,'certificate/followup-v710/benchmark.json')
    summary['campaigns'][machine]=[]
    for mode in ('cpu','auto'):
        run=read(machine,'certificate/campaign-'+mode+'/report.json')
        solves=[json.loads(row) for row in raw(machine,'certificate/campaign-'+mode+'/native-solves.jsonl').decode().splitlines()]
        assert run['best']['official']['ok'] and run['best']['independent']['ok']
        assert len(solves)==36 and all(s['status']=='converged' for s in solves)
        route_records=[]
        for member in opened[machine].getmembers():
            if member.name.startswith('certificate/campaign-'+mode+'/candidates/') and member.name.endswith('/route_summary.json'):
                route=read(machine,member.name);assert route['certified']
                backends={leg['certification_backend'] for leg in route['legs']}
                assert backends=={'cuda' if mode=='auto' else 'cpu'}
                route_records.append(dict(legs=len(route['legs']),backend=next(iter(backends))))
        assert len(route_records)==2
        stage=next(s for s in report['stages'] if s['name']=='campaign-'+mode)
        summary['campaigns'][machine].append(dict(mode=mode,process_seconds=stage['seconds'],campaign_seconds=run['seconds'],best=run['best'],native_solves=len(solves),native_seconds=sum(s['seconds'] for s in solves),routes=route_records,screening=run['screening_telemetry']))
    for name in ('report.json','pytest.log','pipeline-gpu.log','memcheck.log','racecheck.log','synccheck.log','source-manifest.json'):
        extract(machine,'certificate/'+name,machine+'/'+name)
    for name in ('report.json','pytest.log','benchmark.json','benchmark.log','worker.log','worker-v711.py','source-manifest.json'):
        extract(machine,'certificate/followup-v710/'+name,machine+'/followup/'+name)
    for name in ('profile.json','profile.txt','campaign.pstats','worker.py'):
        extract(machine,'profile/'+name,machine+'/profile/'+name)
for name in ('Result.txt','viewer/manifest.json','viewer/trajectories.json'):
    extract('h100','certificate/campaign-auto/best/'+name,'h100-best/'+name)
extract('h100','certificate/campaign-auto/report.json','h100-best/campaign-report.json')
names=['src/spacepdhcg/gtoc12/'+name+'.py' for name in ('cli','gpu_verifier','low_thrust','pipeline')]
names+=['tests/'+name+'.py' for name in ('test_gtoc12_certificate_backend','test_gtoc12_gpu_scvx','test_gtoc12_gpu_verifier','test_gtoc12_gpu_cli')]
for name in names:
    prefix='certificate/followup-v710/repo/' if name in ('src/spacepdhcg/gtoc12/cli.py','tests/test_gtoc12_gpu_cli.py') else 'certificate/repo/'
    data=Path(name).read_bytes()
    assert all(data==raw(machine,prefix+name) for machine in archives),name
    summary['sources'][name]=hashlib.sha256(data).hexdigest()
(out/'summary.json').write_text(json.dumps(summary,indent=2))
(out/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n*.pstats -diff\n')
workers=out/'workers';workers.mkdir()
for path in Path('build/performance').glob('*.py'):
    if any(term in path.name for term in ('profile_v708','campaign_v708','certificate_v709','certificate_v710','certificate_v712','certificate_followup_v710','certificate_followup_v711')):
        shutil.copyfile(path,workers/path.name)
print(json.dumps(dict(out=str(out),campaigns={m:[dict(mode=r['mode'],seconds=r['process_seconds'],score=r['best']['score_kg']) for r in runs] for m,runs in summary['campaigns'].items()})))
