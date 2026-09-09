"""Verify archived bytes, requests, prefix reuse and saved accounting; no physics replay."""
from pathlib import Path
from decimal import Decimal as D,getcontext
import hashlib,io,json,tarfile,zipfile
ROOT=Path(__file__).resolve().parent
getcontext().prec=65
def sha(raw):return hashlib.sha256(raw).hexdigest()
manifest=json.loads((ROOT/'manifest.json').read_text())
assert sha((ROOT/'evidence.tar.gz').read_bytes())==manifest['archive_sha256']
data={}
with tarfile.open(ROOT/'evidence.tar.gz','r|gz') as tar:
    for member in tar:
        assert member.isfile() and member.name in manifest['files'] and member.name not in data
        raw=tar.extractfile(member).read();item=manifest['files'][member.name]
        assert len(raw)==item['bytes'] and sha(raw)==item['sha256'],member.name
        data[member.name]=raw
assert set(data)==set(manifest['files'])
def name(version):
    return {851:'spacepdhcg-family-refinement-v851',852:'spacepdhcg-family-penalty-v852',853:'spacepdhcg-family-retiming-v853',854:'spacepdhcg-family-retiming-v854',855:'spacepdhcg-family-returns-v855'}[version]
def read(version,path):return json.loads(data[name(version)+'/'+path])
def arrays(version,path):
    with zipfile.ZipFile(io.BytesIO(data[name(version)+'/'+path])) as z:
        return {n:z.read(n) for n in z.namelist()}
reports={v:read(v,'report.json') for v in range(851,856)}
assert all(r['complete'] for r in reports.values())
assert all(r['success'] for v,r in reports.items() if v!=853)
assert not reports[853]['success'] and reports[853]['native_solves']==0
assert 'prescribed cargo violates mining production' in reports[853]['error']
assert json.loads(data['terminal/terminal.json'])['all_owned_workers_exited']
base=reports[851]['settings']
assert base['defect_tolerance']==5e-9 and base['virtual_weight']==10000.
for v in (854,855):assert reports[v]['settings']==base
assert sha(data['qoco-runtime/libqoco.so'])==reports[851]['qoco_sha256']=='3df0f33931bcd14d683acbca0781a34ae1d6c237e7840c8b8ffaf358784a8239'
for v in (851,852,854,855):
    assert reports[v]['core_sha256']=='a2ec27ce1b20ec472b0044a5c554751cb704b496f479aff487a852b7cda0527d'
    assert reports[v]['qoco_sha256']==reports[851]['qoco_sha256']
    assert all(not a['certified'] and not a.get('verified_gain',False) for a in reports[v].get('attempts',[]))
for penalty in (100000,1000000):
    settings=read(852,f'{penalty}/settings.json')
    assert settings==dict(base,virtual_weight=float(penalty))
assert data[name(852)+'/boundary.json']==data[name(851)+'/solve-002/boundary.json']
requests={v:read(v,'requests.json') for v in (851,854,855)}
def request_digest(plan):
    x=dict(flights=[(int(l['from']),int(l['to']),float(l['t0']),float(l['tf'])) for l in plan['legs'] if l['role']!='camp'],
        deploy=sorted((int(k),float(v)) for k,v in plan['deploy_epochs'].items()),
        collect=sorted((int(k),float(v)) for k,v in plan['collect_epochs'].items()),
        cargo=sorted((int(k),float(v)) for k,v in plan['collected_mass_kg'].items()))
    return sha(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
for records in requests.values():
    for r in records:assert request_digest(r['plan'])==r['request_sha256']
original=requests[851][0]['plan']
changed={8846,49900,37385,8123,1122}
for rank,delay in enumerate((15,30)):
    plan=requests[854][rank]['plan']
    assert plan['collect_epochs']==original['collect_epochs']
    for body,t in original['deploy_epochs'].items():
        assert plan['deploy_epochs'][body]==t+(delay if int(body) in changed else 0)
        mass=D(str(plan['collected_mass_kg'][body]))
        old=D(str(original['collected_mass_kg'][body]))
        expected=old-D(delay)*D(10)/D('365.25') if int(body) in changed else old
        assert abs(mass-expected)<D('1e-12')
    # The new third-hop duration consumes only a later wait.
    for i,(old,new) in enumerate(zip(original['legs'],plan['legs'])):
        assert (old['from'],old['to'],old['role'])==(new['from'],new['to'],new['role'])
        expected0=old['t0']+(delay if 3<=i<=7 else 0)
        expected1=old['tf']+(delay if 2<=i<=6 else 0)
        assert (new['t0'],new['tf'])==(expected0,expected1),(rank,i)
retimed=requests[854][1]['plan']
for rank,delay in enumerate((60,120,180)):
    plan=requests[855][rank]['plan']
    assert plan['collected_mass_kg']==retimed['collected_mass_kg']
    assert plan['deploy_epochs']==retimed['deploy_epochs'] and plan['collect_epochs']==retimed['collect_epochs']
    assert plan['legs'][:-1]==retimed['legs'][:-1]
    expected=dict(retimed['legs'][-1],tf=retimed['legs'][-1]['tf']+delay)
    assert plan['legs'][-1]==expected
# Compare complete saved .npy payloads for each reused post-clamp prefix.
for v,ranks,count,source_v,source_rank in [(851,(1,2),3,851,0),(854,(0,1),2,851,0),(855,(0,1,2),16,854,1)]:
    for rank in ranks:
        for leg in range(count):
            assert arrays(v,f'rank-{rank:03d}/leg-{leg:02d}.npz')==arrays(source_v,f'rank-{source_rank:03d}/leg-{leg:02d}.npz'),(v,rank,leg)
native=[];certificates=[];rows=[]
for v,r in reports.items():
    prefix=name(v)+'/'
    solution_paths=[k[len(prefix):] for k in data if k.startswith(prefix) and k.endswith('/solution.json') and (k[len(prefix):].startswith('solve-') or v==852)]
    assert len(solution_paths)==r['native_solves']
    solutions=[read(v,k) for k in solution_paths]
    native.extend(solutions)
    legs=[read(v,k[len(prefix):]) for k in data if k.startswith(prefix+'rank-') and '/leg-' in k and k.endswith('.json')]
    certs=[l for l in legs if l['certificate'] is not None]
    certificates.extend(certs)
    for l in certs:assert l['certified'] and l['certification_backend']=='cuda'
    rows.append(dict(version=v,native_solves=len(solutions),cached_solve_readbacks=r.get('cache_hits',0),
        outer_updates=sum(s['iterations'] for s in solutions),accepted_updates=sum(s['accepted_iterations'] for s in solutions),
        flight_certificate_calls=len(certs),complete_verified_routes=0,worker_seconds=r['seconds'],
        attempts=[dict(rank=a['rank'],legs=a['legs'],certified_legs=a['certified_legs']) for a in r.get('attempts',[])]))
assert len(native)==38 and len(certificates)==86
assert sum(r.get('fleet_checks',0) for r in reports.values())==0
unique_qualified=sum(s['status'] in ('converged','iteration_limit') and s['max_defect']<=5e-9 for s in native)
assert unique_qualified==30
result=dict(passed=True,archive_sha256=manifest['archive_sha256'],verified_files=len(data),campaigns=rows,
    unique_native_solves=len(native),unique_numerically_qualified_flight_solutions=unique_qualified,
    flight_certificate_calls_including_rechecks=len(certificates),fleet_checks=0,new_verified_routes=0,
    summed_native_outer_updates=sum(s['iterations'] for s in native),summed_accepted_updates=sum(s['accepted_iterations'] for s in native),
    baseline_third_leg_defect=read(851,'solve-002/solution.json')['max_defect'],penalty_probes=reports[852]['probes'],
    return_failures=[dict(version=v,rank=a['rank'],defect=a['failures'][0].get('solution',{}).get('max_defect'),diagnostic=a['failures'][0]['diagnostic']) for v in (854,855) for a in reports[v]['attempts']],
    score_unchanged=True,physics_replayed_by_this_audit=False)
(ROOT/'saved-audit.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
print(json.dumps(result,indent=2))
