from pathlib import Path
import json,hashlib,os,tarfile,subprocess,shutil
p=Path('build/performance');tag='retained-replay-v514'
r=(p/'run_workspace_pool_v491.py').read_text().replace('workspace-pool-v491',tag).replace('/home/ubuntu/spacepdhcg-early-graph-v471/repo','/home/ubuntu/spacepdhcg-resident-options-v507/repo')
r=r.replace("env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='0'", "env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='0'\n env['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY']='1'")
r=r.replace("tests=['", "tests=['tests/test_gtoc12_gpu_retained_replay.py','tests/test_gtoc12_verifier_knots.py','tests/test_gtoc12_verifier.py','tests/test_gtoc12_gpu_verifier.py','tests/test_gtoc12_gpu_resident_options.py','tests/test_gtoc12_gpu_collection.py','tests/test_gtoc12_gpu_elements.py','")
r=r.replace("run('pytest',[py,'-c',boot,*tests,'-s','-q'],300)", "run('pytest',[py,'-c',boot,*tests,'-s','-q'],600)")
a=r.index(" binary=str(root/");b=r.index(' cli=',a);r=r[:a]+r[b:]
r=r.replace("for name,candidate in [('default',True)]:", "for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:\n  env['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY']='1' if candidate else '0'\n  env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'\n  env['SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY_TRACE']='1'\n  (root/name).mkdir()\n  env['SPACEPDHCG_PHASE_OUTPUT']=str(root/name)")
a=r.index('  cmd=[py,');b=r.index("]+['gtoc12'",a)
r=r[:a]+"  cmd=[py,'-c',cli"+r[b:]
(p/('run_'+tag.replace('-','_')+'.py')).write_text(r)
sources=set(json.loads((p/'resident-options-v506/report.json').read_text())['source_sha256'])|{'tests/test_gtoc12_gpu_retained_replay.py','build/performance/solver_phase_details.py'}
manifest={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sorted(sources)}
mp=p/(tag+'-source-sha256.json');mp.write_text(json.dumps(manifest,indent=2))
# Keep local comparison input sources even if subsequent implementation evolves.
local=p/'retained-replay-campaign-v513'
for f in sorted(sources):
 dest=local/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
shutil.copy2(p/'run_retained_replay_v513.py',local/'run.py')
archive=Path('/tmp/'+tag+'.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for f in sorted(sources):t.add(f,arcname=f)
 t.add(mp,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-"+tag+"');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(r)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_retained_replay_v514.py').write_text(launch)
