from pathlib import Path
import ast
module=ast.parse(Path('build/performance/launch_collect_resident_v297.py').read_text())
source=next(ast.literal_eval(n.value.args[0]) for n in module.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='write_text')
source=source.replace('/home/ubuntu/spacepdhcg-collect-resident-v297','/home/ubuntu/spacepdhcg-collect-resident-v298')
source=source.replace(" cmake=", " run('git-init',['git','init','-q'])\n run('git-add',['git','add','-f','cpp','src','tests','scripts','pyproject.toml'])\n run('git-snapshot',['git','-c','user.name=GPU snapshot','-c','user.email=snapshot@localhost','commit','-qm','Resident collection tables candidate'])\n cmake=")
launch="from pathlib import Path\nimport subprocess\nroot=Path('/home/ubuntu/spacepdhcg-collect-resident-v298');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(source)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(p.pid)\n"
Path('build/performance/launch_collect_resident_v298.py').write_text(launch)
