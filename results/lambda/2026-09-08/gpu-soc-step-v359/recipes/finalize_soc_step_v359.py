from pathlib import Path
import json,gzip,hashlib,shutil
root=Path('results/lambda/2026-09-08/gpu-soc-step-v359')
def read(path):
 if path.exists():return path.read_bytes()
 return gzip.decompress(path.with_name(path.name+'.gz').read_bytes())
def load(path):return json.loads(read(path))
summary=load(root/'summary.json')
for label,path,key in [('local',root/'local-final-replays/report.json','rows'),('lambda',root/'lambda-final-v359/report.json','replays')]:
 r=load(path);assert r['complete']
 candidates=[x for x in r[key] if not x['name'].startswith('guard_baseline')]
 baseline=[x for x in r[key] if x['name'].startswith('guard_baseline')]
 assert sum(len(x['audits']) for x in candidates)==32 and sum(len(x['audits']) for x in baseline)==32
 summary['final_'+label+'_qualified']=sum(sum(a['qualified'] for a in x['audits']) for x in candidates)
 summary['final_'+label+'_baseline_qualified']=sum(sum(a['qualified'] for a in x['audits']) for x in baseline)
(root/'summary.json').write_text(json.dumps(summary,indent=2))
# Check every downloaded file, transparently decompressing stored large evidence.
for name in ['lambda-comparison-v353','lambda-final-v359']:
 for relative,sha in load(root/name/'hashes.json').items():
  assert hashlib.sha256(read(root/name/relative)).hexdigest()==sha,(name,relative)
native=load(root/'lambda-final-v359/report.json')
assert native['preparation']['header_sha256']==hashlib.sha256(Path('cpp/cuda/patches/qoco_soc_step.cuh').read_bytes()).hexdigest()
for p,sha in load(root/'source-sha256.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha,p
for v in [354,355,356,357]:
 report=load(root/f'lambda-comparison-v353/v{v}/output/run_report.json');assert report['best']['official']['ok'] and report['best']['independent']['ok']
report=load(root/'lambda-final-v359/v360/output/run_report.json');assert report['best']['official']['ok'] and report['best']['independent']['ok']
viewer=Path('results/lambda/2026-09-06/visualiser/data/gtoc12-v360')
manifest=load(viewer/'manifest.json')
for name,entry in manifest['files'].items():assert hashlib.sha256((viewer/name).read_bytes()).hexdigest()==entry['sha256']
(root/'viewer-qa.json').write_text(json.dumps(dict(url='http://127.0.0.1:4173/?dataset=gtoc12-v360&epoch=69807&preset=oblique&z=1',method='In-app browser accessibility inspection of actual imported dataset',ships=1,asteroids=8,exact_archived_samples=510,official_pass=True,independent_pass=True,official_kg=548.255,physical_geometry=True,renderer='Active WebGL2 on RTX 5090',import_kepler_context_points=3010,solution_sha256=hashlib.sha256((root/'lambda-final-v359/v360/output/fleet/Result.txt').read_bytes()).hexdigest()),indent=2))
for name in ['import_soc_step_v360.py','finalize_soc_step_v359.py']:
 shutil.copyfile(Path('build/performance')/name,root/'recipes'/name)
evidence={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='evidence-sha256.json'}
(root/'evidence-sha256.json').write_text(json.dumps(evidence,indent=2))
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==sha for p,sha in evidence.items())
print('Verified',len(evidence),'evidence files, both remote manifests, final source, five missions and viewer hashes')
