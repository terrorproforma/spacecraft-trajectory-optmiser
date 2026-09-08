from pathlib import Path
import json,hashlib,tarfile,os,subprocess
base=Path('build/performance/run_stationary_v412.py').read_text()
prefix=base[:base.index(" run('configure'")]
prefix=prefix.replace('spacepdhcg-stationary-v412','spacepdhcg-paired-ephemerides-v417').replace('spacepdhcg-stationary-v408/repo','spacepdhcg-stationary-v412/repo').replace('/tmp/stationary-v412.tar.gz','/tmp/paired-ephemerides-v417.tar.gz').replace('stationary-source-sha256.json','paired-source-sha256.json').replace("core=root/'core-build/cuda/libspacepdhcg_cuda.so'","core=Path('/home/ubuntu/spacepdhcg-stationary-v412/core-build/cuda/libspacepdhcg_cuda.so')")
suffix=''' report['runtime_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]};save()
 boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
 tests=['tests/test_gtoc12_gpu_elements.py','tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_search2.py']
 run('pytest',[py,'-c',boot,*tests,'-q'],300)
 for name in ['memcheck','synccheck']:
  run(name,['/usr/local/cuda/bin/compute-sanitizer','--tool',name,'--error-exitcode','99',py,'-c',boot,'tests/test_gtoc12_gpu_elements.py','-q'],300)
 cli="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('spacepdhcg',run_name='__main__')"
 for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:
  env['SPACEPDHCG_TEST_GTOC12_PAIRED_EPHEMERIDES']='1' if candidate else '0'
  cmd=[py,'-c',cli,'gtoc12','run','--run-id','paired417_'+name,'--output',str(root/name/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','600','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
  run(name,cmd,900)
  r=json.loads((root/name/'output/run_report.json').read_text());assert r['best']['accepted'] and r['best']['official']['ok'] and r['best']['independent']['ok']
  report['campaigns'].append(dict(name=name,candidate=candidate,seconds=r['wall_seconds_total'],score=r['best']['independent']['weighted_score_fixed_bonus_kg'],screening=r['screening']));save()
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print('complete',report['complete'],report.get('error'),flush=True)
'''
runner=prefix+suffix;Path('build/performance/run_paired_ephemerides_v417.py').write_text(runner)
files=['src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/lambert.py','src/spacepdhcg/gtoc12/search.py','tests/test_gtoc12_gpu_elements.py']
manifest=Path('build/performance/paired-source-sha256-v417.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/paired-ephemerides-v417.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='paired-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/paired-ephemerides-v417.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-paired-ephemerides-v417');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(runner)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_paired_ephemerides_v417.py').write_text(program)
print('Uploaded',hashlib.sha256(archive.read_bytes()).hexdigest())
