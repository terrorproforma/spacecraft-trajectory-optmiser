from pathlib import Path
p=Path('build/performance')
s=(p/'prepare_scaled_pool_v541.py').read_text().replace('scaled-pool-v541','scaled-pool-v545').replace('scaled_pool_v541','scaled_pool_v545')
before=" core_flags=['-G'"
after=""" run('git-init',['git','init'])
 run('git-add',['git','add','.'])
 run('git-freeze',['git','-c','user.name=GPU validation','-c','user.email=gpu-validation@localhost','commit','-m','Freeze scaled workspace validation source'])
 report['frozen_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();save()
 core_flags=['-G'"""
assert before in s;s=s.replace(before,after)
(p/'prepare_scaled_pool_v545.py').write_text(s)
