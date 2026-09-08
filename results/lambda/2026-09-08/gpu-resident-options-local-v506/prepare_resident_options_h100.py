from pathlib import Path
import json,hashlib,os,tarfile,subprocess
p=Path('build/performance');tag='resident-options-v497'
run=(p/'run_workspace_pool_v491.py').read_text().replace('workspace-pool-v491',tag).replace('/home/ubuntu/spacepdhcg-early-graph-v471/repo','/home/ubuntu/spacepdhcg-workspace-pool-v491/repo')
run=run.replace("tests=['", "tests=['tests/test_gtoc12_gpu_resident_options.py','tests/test_gtoc12_gpu_collection.py','tests/test_gtoc12_gpu_elements.py','")
run=run.replace("root/'workspace-pool-probe'", "root/'resident-options-probe'").replace('cpp/cuda/tests/gtoc12_workspace_reuse_test.cu','cpp/cuda/tests/gtoc12_resident_options_test.cu')
run=run.replace(' cli=', " run('replay',[binary,str(repo/'build/performance/resident-options-fixture.bin')],300)\n cli=")
run=run.replace("for name,candidate in [('default',True)]:", "for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:\n  env['SPACEPDHCG_TEST_GTOC12_RESIDENT_OPTIONS']='1' if candidate else '0'")
run=run.replace("  report['campaigns'].append", "  screening=r['screening'];assert bool(screening.get('resident_option_builds',0))==candidate\n  if candidate:assert screening.get('resident_option_read_bytes',0)==0 and screening['collection_option_upload_bytes']==0\n  report['campaigns'].append")
(p/'run_resident_options_v497.py').write_text(run)
sources=list(dict.fromkeys(json.loads((p/'workspace-pool-source-sha256-v491.json').read_text()).keys()|set(json.loads((p/'resident-options-sources.json').read_text()))|{'cpp/cuda/tests/gtoc12_resident_options_test.cu','build/performance/resident-options-fixture.bin'}))
manifest={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sorted(sources)}
mp=p/'resident-options-source-sha256-v497.json';mp.write_text(json.dumps(manifest,indent=2))
archive=Path('/tmp/resident-options-v497.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for f in sorted(sources):t.add(f,arcname=f)
 t.add(mp,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=180)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-resident-options-v497');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(run)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_resident_options_v497.py').write_text(launch)
(p/'status_resident_options_v497.py').write_text((p/'status_workspace_pool_v491.py').read_text().replace('workspace-pool-v491','resident-options-v497'))
