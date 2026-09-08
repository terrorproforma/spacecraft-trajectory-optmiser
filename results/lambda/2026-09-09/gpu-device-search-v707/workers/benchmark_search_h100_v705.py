from pathlib import Path
import json
import subprocess
root=Path('/home/ubuntu/spacepdhcg-joint-search-v702')
assert json.loads((root/'report.json').read_text())['complete']
out=root/'followup-v705';out.mkdir(exist_ok=False)
worker=(root/'worker.py').read_text().split('save()\ntry:')[0]
worker=worker.replace("root=Path(__file__).resolve().parent;repo=root/'repo'","root=Path('/home/ubuntu/spacepdhcg-joint-search-v702');repo=root/'repo';out=root/'followup-v705'")
worker=worker.replace("root/'report.json'","out/'report.json'").replace("root/(name+'.log')","out/(name+'.log')")
worker+='''
save()
try:
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
        run('benchmark',[py,'-c',boot,str(root/'benchmark.py'),str(out/'benchmark.json')])
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
'''
(out/'worker.py').write_text(worker)
with (out/'worker.log').open('x') as log:child=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(out/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,root=str(out))))
