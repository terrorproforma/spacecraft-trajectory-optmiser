"""Audit retained source sections and reported score; no numerical propagation."""
from pathlib import Path
import hashlib,json,math,tarfile

root=Path(__file__).resolve().parent;repo=root.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def parts(data):
    result={}
    for line in data.splitlines(keepends=True):
        row=line.split();assert row
        result.setdefault(int(row[0]),bytearray()).extend(line)
    return {k:bytes(v) for k,v in result.items()}
plan=read(root/'plan.json');digest=sha(root/'Result.txt');assert digest==plan['result_sha256']
for p,h in plan['sources'].items():assert sha(repo/p)==h,p
base=parts((repo/'results/lambda/2026-09-09/gpu-collect-reuse-v806/h100-best/Result.txt').read_bytes())
old=parts((repo/'results/lambda/2026-09-09/gpu-regeneration-v799/h100-best/Result.txt').read_bytes())
corrected=parts((repo/'results/local/2026-09-09/current-fleet-composition-v628/Result.txt').read_bytes())
actual=parts((root/'Result.txt').read_bytes());assert set(actual)==set(range(1,24))
assert [s for s in actual if actual[s]!=old[s]]==[4,8]
assert actual[8]==corrected[8] and base[8]==old[8]
assert all(actual[s]==base[s] for s in actual if s!=8)
receipt=read(root/'h100-receipt.json');assert sha(root/'h100.tar.gz')==receipt['sha256']
with tarfile.open(root/'h100.tar.gz') as archive:
    names=[m.name for m in archive.getmembers()];assert len(names)==len(set(names))
    for m in archive.getmembers():
        assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        p=root/('Result.txt' if m.name=='Result.txt' else 'h100/'+m.name)
        assert p.read_bytes()==archive.extractfile(m).read(),m.name
with tarfile.open(root.parent/'gpu-collect-reuse-v806/h100.tar.gz') as archive:
    bonus_bytes=archive.extractfile('runtime/bonus_coefficients.txt').read()
    manifest=json.load(archive.extractfile('FILES.json'))
    assert hashlib.sha256(bonus_bytes).hexdigest()==manifest['runtime/bonus_coefficients.txt']['sha256']
bonus=[float(line.split()[0]) for line in bonus_bytes.decode().splitlines() if line.strip()];assert len(bonus)==60000
mass=[];weighted=[];footprints={};pending=None
for line in (root/'Result.txt').read_text().splitlines():
    f=line.split();ship,event=map(int,f[:2]);footprints.setdefault(ship,set())
    if event>0:footprints[ship].add(event)
    if event==-1:continue
    value=float(f[-1])
    if pending is None:pending=(ship,event,value)
    else:
        assert pending[:2]==(ship,event)
        if event>0 and value>pending[2]:
            cargo=value-pending[2];mass.append(cargo);weighted.append(cargo*bonus[event-1])
        pending=None
assert pending is None and len(set.union(*footprints.values()))==199
assert all(not footprints[i]&footprints[j] for i in footprints for j in footprints if i<j)
raw=math.fsum(mass);score=math.fsum(weighted);limit=min(100.,2*math.exp(.004*raw/23));assert limit>=23
for side in ('local','h100'):
    r=read(root/side/'report.json')
    assert r['complete'] and r['success'] and r['qualified'] and r['gpu_solves']==0
    assert r['independent']['ok'] and r['official']['ok'] and r['solution_sha256']==digest
    assert r['independent']['ships']==r['official']['ships']==23
    assert r['independent']['mined_asteroids']==r['official']['mined_asteroids']==199
    assert abs(r['score_kg']-score)<1e-8 and abs(r['total_mass_kg']-raw)<1e-8
assert abs(plan['expected_weighted_kg']-score)<1e-8 and abs(plan['expected_raw_kg']-raw)<1e-8
viewer=read(root/'h100/viewer/manifest.json')
assert viewer['source']['sha256']==digest
for name,item in viewer['files'].items():assert sha(root/'h100/viewer'/name)==item['sha256']
print(json.dumps(dict(result_sha256=digest,weighted_kg=score,raw_kg=raw,ships=23,asteroids=199,ship_limit=limit,changed_ships=[4,8],retained_sections=21,both_checkers_pass_both_hosts=True),indent=2))
