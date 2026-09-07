from pathlib import Path
import subprocess,shutil,tarfile,json,hashlib,os,time
root=Path('/home/ubuntu/spacepdhcg-gpu-execution-v328');repo=root/'repo'
report=dict(pid=os.getpid(),complete=False,rows=[])
def save(): (root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
 shutil.copytree('/home/ubuntu/spacepdhcg-conic-retry-v314/repo',repo,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
 with tarfile.open('/tmp/gpu-execution-v328.tar.gz') as t:
  for m in t.getmembers():
   target=(repo/m.name).resolve();assert target.is_relative_to(repo.resolve()) and m.isfile()
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(m).read())
 manifest=json.loads((repo/'execution-source-sha256.json').read_text())
 assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
 report['source_sha256']=manifest;save()
 template=Path('/home/ubuntu/spacepdhcg-retry-campaign-v323/run.py').read_text()
 # Extract the same environment setup without executing an old campaign.
 setup=template[template.index('env={'):template.index('boot=')]
 scope=dict(Path=Path,os=os,ast=__import__('ast'),repo=repo,integrated=Path('/home/ubuntu/spacepdhcg-conic-retry-v314'))
 exec(setup,scope);env=scope['env'];env['SPACEPDHCG_GTOC12_GPU_TESTS']='1'
 boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
 import fcntl
 with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with (root/'pytest.log').open('x') as log:
   test=subprocess.run([scope['python'],'-c',boot,'tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_gpu_scvx.py','-q'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
  report['pytest_returncode']=test.returncode;save();test.check_returncode()
 for i,mode in enumerate(['graph','dispatch','dispatch','graph']):
  version=329+i;path=root/('v'+str(version));path.mkdir(exist_ok=False)
  source=template.replace('/home/ubuntu/spacepdhcg-retry-campaign-v323',str(path)).replace('gpu_conic_retry_v323','gpu_execution_'+mode+'_v'+str(version))
  source=source.replace("repo=integrated/'repo'","repo=Path("+repr(str(repo))+")")
  source=source.replace("report['kernel_source_sha256']=", "cmd += ['--gpu-execution',"+repr(mode)+"]\nreport['execution_source_sha256']=json.loads((repo/'execution-source-sha256.json').read_text())\nreport['kernel_source_sha256']=",1)
  (path/'run.py').write_text(source)
  with (path/'runner.log').open('x') as log:
   child=subprocess.Popen(['python3',str(path/'run.py')],stdout=log,stderr=subprocess.STDOUT)
   row=dict(version=version,mode=mode,pid=child.pid,complete=False);report['rows'].append(row);save()
   row['returncode']=child.wait(timeout=1900)
  run=json.loads((path/'report.json').read_text());assert run['complete'] and run['returncode']==0
  output=json.loads((path/'output/run_report.json').read_text())
  row.update(complete=True,seconds=output['wall_seconds_total'],score=output['best']['score_kg'],official=output['best']['official']['ok'],independent=output['best']['independent']['ok'],selected=output['gpu_execution_selected'])
  assert row['selected']==mode
  save();print(row,flush=True)
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print('complete',report['complete'],report.get('error'),flush=True)
