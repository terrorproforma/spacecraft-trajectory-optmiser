from pathlib import Path
import ast,json,hashlib,os,tarfile,subprocess
p=Path('build/performance')
sources=json.loads((p/'resident-options-sources.json').read_text())+['cpp/cuda/tests/gtoc12_resident_options_test.cu']
run=(p/'check_workspace_pool_v490.py').read_text().replace('workspace-pool-v490','resident-options-v498')
a=run.index('sources=');b=run.index('\nfor name in sources:',a);old=ast.literal_eval(run[a+8:b]);sources=list(dict.fromkeys(old+sources));run=run[:a]+'sources='+repr(sources)+run[b:]
a=run.index('jobs=');b=run.index('\nr=dict',a);jobs=run[a:b]
jobs=jobs[jobs.index("[('pytest'"):];jobs="jobs="+jobs
jobs=jobs.replace("boot,'tests", "boot,'tests/test_gtoc12_gpu_resident_options.py','tests/test_gtoc12_gpu_collection.py','tests/test_gtoc12_gpu_elements.py','tests",1)
run=run[:a]+jobs+run[b:]
run=run.replace(" r['campaign']=", " assert result['screening'].get('resident_option_read_bytes',0)==0 and result['screening']['collection_option_upload_bytes']==0\n assert result['screening']['resident_option_selection_download_bytes']==40*(result['screening']['completed_collection_queries']+result['screening']['completed_return_feasibility_queries'])\n r['campaign']=")
(p/'check_resident_options_v498.py').write_text(run)
fleet=(p/'run_workspace_pool_fleet_v488.py').read_text().replace('workspace-pool-fleet-v488','resident-options-fleet-v499').replace('workspace_pool_fleet488','resident_options_fleet499').replace('build-spacepdhcg-workspace-pool-v482','build-spacepdhcg-resident-options-v498')
fleet=fleet.replace("env['SPACEPDHCG_TEST_GTOC12_QOCO_POOL']='1'",'')
a=fleet.index('for p in [')+len('for p in ');b=fleet.index(']},pid',a)+1;names=list(dict.fromkeys(ast.literal_eval(fleet[a:b])+sources));fleet=fleet[:a]+repr(names)+fleet[b:]
(p/'run_resident_options_fleet_v499.py').write_text(fleet)
remote=(p/'run_resident_options_v497.py').read_text().replace('resident-options-v497','resident-options-v500').replace('/home/ubuntu/spacepdhcg-workspace-pool-v491/repo','/home/ubuntu/spacepdhcg-resident-options-v497/repo')
remote=remote.replace("core=root/'core-build/cuda/libspacepdhcg_cuda.so'", "core=Path('/home/ubuntu/spacepdhcg-resident-options-v497/core-build/cuda/libspacepdhcg_cuda.so')")
a=remote.index(" run('configure'");b=remote.index(" report['runtime_sha256']",a);remote=remote[:a]+remote[b:]
a=remote.index(" binary=str(");b=remote.index(' cli=',a);remote=remote[:a]+remote[b:]
remote=remote.replace("for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:\n  env['SPACEPDHCG_TEST_GTOC12_RESIDENT_OPTIONS']='1' if candidate else '0'", "for name,candidate in [('default',True)]:")
remote=remote.replace("  report['campaigns'].append", "  assert screening['resident_option_selection_download_bytes']==40*(screening['completed_collection_queries']+screening['completed_return_feasibility_queries'])\n  report['campaigns'].append")
(p/'run_resident_options_v500.py').write_text(remote)
names=list(dict.fromkeys(list(json.loads((p/'resident-options-source-sha256-v497.json').read_text()))+sources))
mp=p/'resident-options-source-sha256-v500.json';mp.write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in names},indent=2))
archive=Path('/tmp/resident-options-v500.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for f in names:t.add(f,arcname=f)
 t.add(mp,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=180)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-resident-options-v500');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(remote)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_resident_options_v500.py').write_text(launch)
(p/'status_resident_options_v500.py').write_text((p/'status_resident_options_v497.py').read_text().replace('resident-options-v497','resident-options-v500'))
