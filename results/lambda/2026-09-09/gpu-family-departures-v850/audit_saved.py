"""Standard-library saved-byte and Decimal cargo audit; executes no archived code."""
from pathlib import Path
from decimal import Decimal as D, getcontext
from collections import defaultdict
import hashlib,json,tarfile

ROOT=Path(__file__).resolve().parent
getcontext().prec=65
def sha(raw):return hashlib.sha256(raw).hexdigest()
manifest=json.loads((ROOT/'manifest.json').read_text())
assert sha((ROOT/'evidence.tar.gz').read_bytes())==manifest['archive_sha256']
payload={}
with tarfile.open(ROOT/'evidence.tar.gz','r|gz') as tar:
    for member in tar:
        assert member.isfile() and member.name in manifest['files'] and member.name not in payload
        raw=tar.extractfile(member).read();item=manifest['files'][member.name]
        assert len(raw)==item['bytes'] and sha(raw)==item['sha256'],member.name
        payload[member.name]=raw
assert set(payload)==set(manifest['files'])
def read(name):return json.loads(payload['continued/'+name])
report=read('report.json');initial=json.loads(payload['initial/report.json'])
assert report['success'] and report['complete'] and initial['complete'] and not initial['success']
assert 'Out of range float values' in initial['error']
assert payload['continued/ship-02-unit/plans.json']==payload['initial/ship-02-unit/plans.json']
assert payload['continued/ship-02-unit/candidates.json']==payload['initial/ship-02-unit/candidates.json']
assert payload['continued/ship-02-screen.json']==payload['initial/ship-02-screen.json']
assert read('terminal.json')['worker_exited'] and read('terminal.json')['initial_worker_exited']
assert report['trajectory_solves']==report['certificates']==report['fleet_checks']==0
assert report['core_sha256']=='a2ec27ce1b20ec472b0044a5c554751cb704b496f479aff487a852b7cda0527d'
assert report['source_manifest_sha256']=='2cbf1221dc2eace0b0d1b10623a629be3daaa985bd1839769a55d306b6500014'
assert sha(payload['continued/fleet.txt'])==report['fleet_sha256']=='1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da'
assert sha(payload['continued/bonus_coefficients.txt'])==report['bonus_sha256']=='e8a3795e599556ed5b66713ab1fa176de93ef37f93cb2a4a87d561539b1caa21'
weights={i+1:D(line.split()[0]) for i,line in enumerate(payload['continued/bonus_coefficients.txt'].decode().splitlines()) if line.strip()}
assert len(weights)==60000
events=[line.split() for line in payload['continued/fleet.txt'].decode().splitlines() if int(line.split()[1])>0]
assert len(events)%2==0
raw=defaultdict(D);weighted=defaultdict(D);occupied=defaultdict(set)
for a,b in zip(events[::2],events[1::2]):
    assert a[:3]==b[:3] and len(a)==len(b)==10
    ship,body=int(a[0]),int(a[1]);delta=D(b[-1])-D(a[-1]);occupied[ship].add(body)
    if delta>0:raw[ship]+=delta;weighted[ship]+=delta*weights[body]
    else:assert delta==D('-40')
assert len(raw)==24
margin=sum(raw.values())-D(24)*(D(12).ln())/D('.004')
assert abs(margin-D(str(report['raw_margin_kg'])))<D('1e-9')
def identity(plan):
    data={k:plan[k] for k in ('deploy_epochs','collect_epochs','collected_mass_kg','foreign_deploy_epochs')}
    data['flights']=[{k:l[k] for k in ('from','to','t0','tf')} for l in plan['legs'] if l['role']!='camp']
    return sha(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
rows=[];qualified={};all_ids=set();known_seconds=0.;known_candidates=0;known_branches=0
for arm in report['routes']:
    ship,mode=arm['ship'],arm['arm'];prefix=f'ship-{ship:02d}-{mode}'
    plans=read(prefix+'/plans.json');saved=read(prefix+'/candidates.json')
    assert len(plans)==len(saved)==arm['candidates']
    excluded=set().union(*(v for k,v in occupied.items() if k!=ship))
    actual=[];raw_ok=0;positive=0;identities=set()
    for rank,(plan,metric) in enumerate(zip(plans,saved)):
        cargo={int(k):D(str(v)) for k,v in plan['collected_mass_kg'].items()}
        r=sum(cargo.values());w=sum(weights[a]*m for a,m in cargo.items())
        delta=w-weighted[ship];budget=margin+r-raw[ship]
        conflicts=sorted(set(plan['asteroids'])&excluded)
        assert conflicts==metric['conflicts']==[]
        for key,value in [('raw_kg',r),('weighted_kg',w),('raw_delta_kg',r-raw[ship]),('weighted_delta_kg',delta),('fleet_raw_margin_kg',budget)]:
            assert abs(D(str(metric[key]))-value)<D('1e-8'),(prefix,rank,key)
        accept=not conflicts and budget>=D('.000001') and delta>D('.1')
        assert accept==metric['qualifies_for_refinement']
        ident=identity(plan);identities.add(ident);all_ids.add(ident)
        positive+=delta>D('.1');raw_ok+=budget>=D('.000001')
        if accept:
            actual.append(rank)
            q=qualified.setdefault(ident,dict(request_identity=ident,ship=ship,weighted_gain_kg=str(delta),raw_gain_kg=str(r-raw[ship]),fleet_raw_margin_kg=str(budget),references=[]))
            q['references'].append(dict(member='continued/'+prefix+'/plans.json',rank=rank))
    assert actual==[c['rank'] for c in arm['refinement_candidates']]
    if arm['search_seconds'] is not None:
        known_seconds+=arm['search_seconds'];known_candidates+=len(plans);known_branches+=arm['lambert_evaluations']
    rows.append(dict(ship=ship,arm=mode,candidates=len(plans),unique_physical_requests=len(identities),
        positive_weighted_forecasts=positive,raw_budget_passes=raw_ok,refinement_ranks=actual,search_seconds=arm['search_seconds']))
assert len(qualified)==3
result=dict(passed=True,archive_sha256=manifest['archive_sha256'],verified_files=len(payload),rows=rows,
    total_candidates=sum(r['candidates'] for r in rows),unique_physical_requests=len(all_ids),
    qualified_forecasts=list(qualified.values()),known_search_seconds=known_seconds,known_search_candidates=known_candidates,
    known_lambert_branches=known_branches,candidates_per_second_for_seven_recorded_searches=known_candidates/known_seconds,
    missing_measurement='First search completed before reporting failed; no invented timer or branch count.',
    fleet_weighted_kg=str(sum(weighted.values())),fleet_raw_kg=str(sum(raw.values())),fleet_raw_margin_kg=str(margin),
    new_trajectory_certificates=0,score_gain_claimed=False)
(ROOT/'saved-audit.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
print(json.dumps(result,indent=2))
