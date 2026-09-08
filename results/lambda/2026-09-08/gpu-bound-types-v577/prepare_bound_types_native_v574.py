from pathlib import Path
p=Path('build/performance');s=(p/'run_bound_types_native_v574.py').read_text()
code="from pathlib import Path\nimport subprocess,json\nrepo=Path('/home/ubuntu/spacepdhcg-bound-types-v569/repo')\n"
code+="path=repo/'build/performance/run_bound_types_native_v574.py';path.write_text("+repr(s)+")\n"
code+="with Path('/home/ubuntu/bound-types-v574-runner.log').open('x') as log:child=subprocess.Popen(['python3',str(path)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid)))\n"
(p/'launch_bound_types_native_v574.py').write_text(code)
