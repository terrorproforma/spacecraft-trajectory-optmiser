from pathlib import Path
import json,hashlib,tarfile,subprocess,os
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out: out.write(Path('traj-key.pem').read_bytes())
p=Path('build/performance')
local=(p/'check_workspace_pool_v482.py').read_text().replace('workspace-pool-v482','workspace-pool-v490')
local=local.replace("sources=[", "sources=['cpp/cuda/tests/gtoc12_workspace_reuse_test.cu',")
local=local.replace("binary='/home/angus/early-graph-probe-v464'", "binary='/home/angus/workspace-pool-probe-v490'").replace("'cpp/cuda/tests/gtoc12_scvx_test.cu','-L'", "'cpp/cuda/tests/gtoc12_workspace_reuse_test.cu','-L'")
local=local.replace("env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='1'", "env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='0'")
a=local.index('jobs=');b=local.index('\nr=dict',a)
cli="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('spacepdhcg',run_name='__main__')"
args=['gtoc12','run','--run-id','workspace_pool_final','--output','OUTPUT','--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','600','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
job="jobs=[('compile',compile_cmd)]+[(tool,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',binary]) for tool in ['memcheck','synccheck','racecheck']]+[('pytest',[py,'-c',boot,'tests/test_gtoc12_gpu_workspace_pool.py','tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_run_final_verification.py','-s','-q']),('default',[py,'-c',"+repr(cli)+"]+"+repr(args).replace("'OUTPUT'", "str(root/'output')")+")]"
local=local[:a]+job+local[b:]
local=local.replace(" r['complete']=True", " result=json.loads((root/'output/run_report.json').read_text());assert result['best']['accepted'] and result['best']['official']['ok'] and result['best']['independent']['ok']\n r['campaign']=dict(cli_seconds=result['wall_seconds_total'],score=result['best']['independent']['weighted_score_fixed_bonus_kg'],screening=result['screening'])\n r['complete']=True")
(p/'check_workspace_pool_v490.py').write_text(local)
remote=(p/'run_workspace_pool_v484.py').read_text().replace('workspace-pool-v484','workspace-pool-v491')
# Tests and probes run once, then a default (unset POOL) campaign.
a=remote.index(' for name,candidate in [');b=remote.index(" report['complete']=True",a)
campaign=" for name,candidate in [('default',True)]:\n  cmd=[py,'-c',"+repr(cli)+"]+"+repr(args).replace("'OUTPUT'", "str(root/name/'output')")+"\n  run(name,cmd,900)\n  r=json.loads((root/name/'output/run_report.json').read_text());assert r['best']['accepted'] and r['best']['official']['ok'] and r['best']['independent']['ok']\n  report['campaigns'].append(dict(name=name,candidate=candidate,seconds=r['wall_seconds_total'],score=r['best']['independent']['weighted_score_fixed_bonus_kg'],screening=r['screening']));save()\n"
remote=remote[:a]+campaign+remote[b:]
remote=remote.replace(" env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'",'')
(p/'run_workspace_pool_v491.py').write_text(remote)
files=list(json.loads((p/'workspace-pool-source-sha256-v484.json').read_text()))
mp=p/'workspace-pool-source-sha256-v491.json';mp.write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},indent=2))
archive=Path('/tmp/workspace-pool-v491.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for f in files:t.add(f,arcname=f)
 t.add(mp,arcname='fused-tables-source-sha256.json')
subprocess.run(['scp','-q','-i','/tmp/traj-key.pem','-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-workspace-pool-v491');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(remote)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_workspace_pool_v491.py').write_text(launch)
(p/'status_workspace_pool_v491.py').write_text((p/'status_workspace_pool_v484.py').read_text().replace('workspace-pool-v484','workspace-pool-v491'))
