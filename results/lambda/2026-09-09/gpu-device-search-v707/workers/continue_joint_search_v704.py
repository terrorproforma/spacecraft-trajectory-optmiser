from pathlib import Path
import json
import os
import subprocess

root=Path('/home/angus/spacepdhcg-joint-search-v702')
report=json.loads((root/'report.json').read_text());assert report['complete']
out=root/'followup-v704';out.mkdir(exist_ok=False)
worker=(root/'worker.py').read_text()
prefix=worker.split('save()\ntry:')[0]
prefix=prefix.replace("root=Path(__file__).resolve().parent;repo=root/'repo'","root=Path('/home/angus/spacepdhcg-joint-search-v702');repo=root/'repo';out=root/'followup-v704'")
prefix=prefix.replace("root/'report.json'","out/'report.json'").replace("root/(name+'.log')","out/(name+'.log')")
body='''
save()
boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    for name,mode,case in [('memcheck-empty','memcheck','empty'),('racecheck','racecheck','moves'),('synccheck','synccheck','moves')]:
        try:run(name,[cuda+'/bin/compute-sanitizer','--tool',mode,'--error-exitcode','86',py,'-c',boot,'-q','tests/test_gtoc12_gpu_joint_search.py','-k','whole_search and '+case])
        except RuntimeError:pass
    try:
        run('benchmark',[py,str(root/'benchmark.py'),str(out/'benchmark.json')])
        report['benchmark_success']=True
    except BaseException:report['error']=traceback.format_exc()
report['success']=all(s['returncode']==0 for s in report['stages']);report['complete']=True;save()
'''
(out/'worker.py').write_text(prefix+body)
with (out/'worker.log').open('x') as log:child=subprocess.Popen(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',str(out/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,root=str(out))))

