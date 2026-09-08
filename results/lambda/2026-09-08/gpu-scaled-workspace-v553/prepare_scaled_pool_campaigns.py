from pathlib import Path
p=Path('build/performance')
s=(p/'run_preserve_campaign_v537.py').read_text().replace('preserve-campaign-v537','scaled-pool-campaign-v543').replace('preserve537','scaled543')
s=s.replace('build-spacepdhcg-retained-replay-v512','build-spacepdhcg-scaled-pool-v539').replace('build-qoco-preserve-objective-v534','build-qoco-scaled-pool-v540')
start=s.index("  env['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1'")
end=s.index("  r['campaigns']=[]",start)
s=s[:start]+s[end:]
s=s.replace("   env['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1' if enabled else '0'", "   env['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1' if enabled else '0'\n   env['SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL']='1' if enabled else '0'")
(p/'run_scaled_pool_campaign_v543.py').write_text(s)
remote=s.replace("root=Path('build/performance/scaled-pool-campaign-v543');root.mkdir(exist_ok=False)","root=Path('/home/ubuntu/spacepdhcg-scaled-pool-campaign-v544')")
for old,new in {
 '/home/angus/build-spacepdhcg-scaled-pool-v539/final':'/home/ubuntu/spacepdhcg-scaled-pool-v541/core-build/cuda',
 '/home/angus/build-qoco-scaled-pool-v540/final':'/home/ubuntu/spacepdhcg-scaled-pool-v541/final',
 '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python':'/home/ubuntu/spacepdhcg/v1/.venv/bin/python',
 '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data':'/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',
 '/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib':'/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib',
 '/usr/local/cuda-12.8':'/usr/local/cuda','/home/angus/.spacepdhcg-gpu.lock':'/home/ubuntu/.spacepdhcg-gpu.lock',
 "subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()":"'dd4f4a5adf668247fa64b840c14f846bf8b37be2'"}.items():remote=remote.replace(old,new)
(p/'run_scaled_pool_campaign_v544.py').write_text(remote)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-scaled-pool-campaign-v544');root.mkdir(exist_ok=False)\n"
launch+="(root/'run.py').write_text("+repr(remote)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],cwd='/home/ubuntu/spacepdhcg-scaled-pool-v541/repo',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_scaled_pool_campaign_v544.py').write_text(launch)
