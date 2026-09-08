from pathlib import Path
import hashlib,json,statistics,shutil,tarfile
p=Path('build/performance');remote=p/'retrieved-parallel-v622';dest=Path('results/lambda/2026-09-09/gpu-parallel-directions-v622')
read=lambda path:json.loads(path.read_text())
summary=dict(policy='Opt-in only: one full warp per Lambert direction for batches <=1024; larger batches retain current dispatch. No production default change.',campaigns={},microbenchmarks={},validation={},incumbent_weighted_kg=12805.194102488575)
for gpu,campaign,validation in [('RTX5090',p/'parallel-directions-campaign-v618',p/'parallel-directions-v616'),('H100',remote/'parallel-directions-campaign-v617',remote/'parallel-directions-v617')]:
 r=read(campaign/'report.json');v=read(validation/'report.json');assert r['complete'] and v['complete'] and not r.get('error') and not v.get('error')
 assert '106 passed' in (validation/'pytest.log').read_text()
 times={mode:[c['process_seconds'] for c in r['campaigns'] if c['candidate']==flag] for mode,flag in [('baseline',False),('parallel',True)]}
 medians={k:statistics.median(x) for k,x in times.items()}
 cases=[]
 for c in r['campaigns']:
  report=read(campaign/c['name']/'output/run_report.json');best=report['best']
  assert best['accepted'] and best['official']['ok'] and best['independent']['ok']
  assert abs(c['score']-548.2546201232)<1e-6
  assert c['screening']['completed_branch_requests']==45188558 and c['screening']['completed_collection_options']==2782091
  cases.append(dict(name=c['name'],process_seconds=c['process_seconds'],weighted_score_kg=c['score']))
 summary['campaigns'][gpu]=dict(cases=cases,seconds=times,medians=medians,less_time_percent=100*(1-medians['parallel']/medians['baseline']),scope='Complete one-ship process, ABBA order, same frozen binary per GPU; ranges overlap.',core_sha256=r['core_sha256'])
 summary['validation'][gpu]=dict(tests=106,lambert_sanitizers=['memcheck','synccheck','racecheck'],source_sha256=v['source_sha256'],core_sha256=v['core_sha256'],qoco_sha256=v['qoco_sha256'])
 for name,sha in v['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
 summary['microbenchmarks'][gpu]=read(validation/'benchmark.json')
summary['rejected_layouts']=dict(half_warp='Does not consistently accelerate small batches; slower at 65536 on both GPUs.',unrestricted_full_warp='Small-batch H100 gains, but large-batch regressions on both GPUs; cutoff protects existing large dispatch.',archive_paths=['split-directions-v605','split-directions-v613','split-directions-v606','split-directions-v612'])
summary['limitations']='No reliable overall speedup established by two runs per mode. No best-fleet improvement or submission. All trajectories require low-thrust certification. Lambert-only sanitizer passes do not resolve full-QOCO/cuDSS sanitizer issues. CPU orchestration remains.'
files={}
for group in ['split-directions-v602','split-directions-v606','split-directions-v612','split-directions-campaign-v607','parallel-directions-v616','parallel-directions-campaign-v618']:
 for path in sorted((p/group).rglob('*')):
  if path.is_file() and '__pycache__' not in path.parts:files[group+'/'+path.relative_to(p/group).as_posix()]=path
for version,kind in [(601,'split'),(603,'split'),(604,'split'),(610,'split'),(615,'parallel')]:
 build=Path(f'/home/angus/spacepdhcg-{kind}-directions-v{version}')
 for name in ['report.json','configure.log','build.log','build/cuda/libspacepdhcg_cuda.so','repo/cpp/cuda/src/orbitweaver_gpu.cu','repo/tests/test_gtoc12_gpu_split_directions.py']:
  if (build/name).is_file():files[f'build-v{version}/'+name]=build/name
files['qoco/libqoco.so']=Path('/home/angus/build-qoco-scaled-pool-v540/final/libqoco.so')
for pattern in ['*split*directions*.py','*parallel*directions*.py','*split*campaign*.py','*parallel*campaign*.py','*parallel*v622.py','restore_split_fixture.py','benchmark_fast_root.py','solver_phase_details.py']:
 for path in p.glob(pattern):files['workers/'+path.name]=path
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
(dest/'local-archive-manifest.json').write_text(json.dumps(manifest,indent=2))
archive=dest/'local-raw.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for name,path in files.items():t.add(path,arcname=name,recursive=False)
with tarfile.open(archive,'r:gz') as t:
 for m in t.getmembers():assert m.isfile() and hashlib.sha256(t.extractfile(m).read()).hexdigest()==manifest[m.name]
summary['local_archive']=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),verified_members=len(manifest))
(dest/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ['archive_parallel_v622.py','collect_parallel_v622.py','publish_parallel_v622.py']:
 shutil.copy2(p/name,dest/name)
(dest/'.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
print(json.dumps(summary['campaigns'],indent=2))
