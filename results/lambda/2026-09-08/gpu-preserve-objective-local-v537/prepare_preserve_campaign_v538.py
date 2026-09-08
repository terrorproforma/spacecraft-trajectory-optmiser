from pathlib import Path
p=Path('build/performance')
s=(p/'run_preserve_campaign_v537.py').read_text()
s=s.replace("root=Path('build/performance/preserve-campaign-v537');root.mkdir(exist_ok=False)","root=Path('/home/ubuntu/spacepdhcg-preserve-campaign-v538')")
s=s.replace('build-qoco-preserve-objective-v534','spacepdhcg-preserve-objective-v535').replace('/home/angus/spacepdhcg-preserve-objective-v535','/home/ubuntu/spacepdhcg-preserve-objective-v535')
s=s.replace('/home/angus/build-spacepdhcg-retained-replay-v512/final','/home/ubuntu/spacepdhcg-retained-replay-v514/core-build/cuda')
s=s.replace('/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','/home/ubuntu/spacepdhcg/v1/.venv/bin/python')
s=s.replace('/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data','/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data')
s=s.replace('/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib','/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib').replace('/usr/local/cuda-12.8','/usr/local/cuda')
s=s.replace('/home/angus/.spacepdhcg-gpu.lock','/home/ubuntu/.spacepdhcg-gpu.lock')
s=s.replace("subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()", "'71fb2b51b798ff0cb8b2b9002f5ed9a056f1de20'")
(p/'run_preserve_campaign_v538.py').write_text(s)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-preserve-campaign-v538');root.mkdir(exist_ok=False)\n"
launch+="(root/'run.py').write_text("+repr(s)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],cwd='/home/ubuntu/spacepdhcg-preserve-objective-v535/repo',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_preserve_campaign_v538.py').write_text(launch)
