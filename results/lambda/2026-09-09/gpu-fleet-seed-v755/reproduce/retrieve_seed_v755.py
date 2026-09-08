from pathlib import Path
import hashlib,json,os,shutil,statistics,subprocess,tarfile
dest=Path('results/lambda/2026-09-09/gpu-fleet-seed-v755');dest.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
for name in ('h100.tar.gz','receipt.json'):
    subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-fleet-seed-evidence-v755/'+name,str(dest/(name if name.endswith('gz') else 'h100-receipt.json'))],check=True,timeout=55)
for name in ('local.tar.gz','receipt.json'):
    shutil.copyfile(Path.home()/'spacepdhcg-fleet-seed-evidence-v755'/name,dest/(name if name.endswith('gz') else 'local-receipt.json'))
summary=dict(verified_score_kg=12842.970672270894,score_change_kg=0,solution_sha256='fba0ee086ae55d6c690f0e5dbaf874834b6f63e2f238e436d90671bdddff3c5f',scope='Parallel CUDA fleet seed construction from existing route records. No new trajectory or mission qualification.',method='Seven rotating interleaved repetitions per mode/cap; medians exclude repeat zero; exact selection, objective, bound and search-count equality.',profile_method='Standalone instrumented fleet libraries with CUDA events; three repetitions per cap/rounds, medians exclude repeat zero. Stage times are not production API timings.')
for side in ('local','h100'):
    archive=dest/(side+'.tar.gz');receipt=json.loads((dest/(side+'-receipt.json')).read_text())
    assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('FILES.json'));names=[m.name for m in tar.getmembers()]
        assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
        for name,e in manifest.items():
            b=tar.extractfile(name).read();assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],name
            if '/repo/' not in name and not name.endswith('.so') and name!='input/pool.json':
                target=dest/side/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
        (dest/(side+'-files.json')).write_text(json.dumps(manifest,indent=2))
        if side=='local':(dest/'pool.json').write_bytes(tar.extractfile('input/pool.json').read())
    r=json.loads((dest/side/'v752/report.json').read_text());m=r['medians'];old=m['0']['old_retained'];new=m['0']['retained']
    stages={}
    for v in ('v746','v748','v751'):
        rows=json.loads((dest/side/v/'report.json').read_text())['rows']
        rows=[x for x in rows if x['cap']==0 and x['rounds']==16 and x['repeat']>0]
        stages[v]={k:statistics.median(x[k] for x in rows) for k in ('seed_ms','search_ms','finish_ms','sum_ms')}
        stages[v].update({k:statistics.median(sum(x[k]) for x in rows) for k in ('evaluate_ms','accept_ms')})
    summary[side]=dict(medians=m,retained_speedup=old['seconds']/new['seconds'],retained_selections_per_second=1/new['seconds'],logical_proposals_per_second=179205/new['seconds'],one_shot_speedup=m['0']['old_one_shot']['seconds']/m['0']['new_one_shot']['seconds'],first_workspace_setup_seconds=r['setup_seconds'],seed_stage_speedup=stages['v746']['seed_ms']/stages['v751']['seed_ms'],profile_stages=stages,input_columns=2492,usable_columns=2488,proposals=179205,moves=2,sweeps=3,tree_nodes_with_200000_budget=35145,differential_cases=128,archive_sha256=receipt['sha256'],verified_members=len(manifest))
(dest/'summary.json').write_text(json.dumps(summary,indent=2))
repro=dest/'reproduce';repro.mkdir()
for name in ('bench_fleet_seed_v749.py','bench_fleet_seed_v752.py','check_seed_v753.py','differential_seed_v754.py','package_seed_v755.py','retrieve_seed_v755.py','profile_fleet_stages_v746.py','profile_fleet_stages_v748.py','profile_fleet_stages_v751.py'):
    shutil.copyfile(Path('build/performance')/name,repro/name)
text=Path('results/lambda/2026-09-09/gpu-fleet-topology-v745/reproduce/replay.py').read_text().replace('v742','v750')
(repro/'replay.py').write_text(text)
(dest/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n')
print(json.dumps(summary,indent=2))
