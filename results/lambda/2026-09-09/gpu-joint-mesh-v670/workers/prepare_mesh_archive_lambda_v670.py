from pathlib import Path
p=Path('build/performance')
archive=(p/'archive_mesh_v670.py').read_text()
worker='''from pathlib import Path
import json,subprocess,time,traceback
root=Path('/home/ubuntu/spacepdhcg-joint-mesh-v666')
record=dict(complete=False,success=False)
try:
 deadline=time.monotonic()+1200
 while time.monotonic()<deadline:
  r=json.loads((root/'campaign-v667/report.json').read_text())
  if r['complete']:break
  time.sleep(5)
 assert r['complete'] and r['success']
 with (root/'archive-v670.log').open('x') as log:
  child=subprocess.run(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'archive-v670.py'),str(root),'/home/ubuntu/joint-mesh-v670-h100.tar.gz','/home/ubuntu/spacepdhcg-scaled-pool-v545/final/libqoco.so'],stdout=log,stderr=subprocess.STDOUT)
 assert child.returncode==0
 record['success']=True
except BaseException:record['exception']=traceback.format_exc()
record['complete']=True;(root/'archive-status-v670.json').write_text(json.dumps(record,indent=2))
'''
code="from pathlib import Path\nimport json,subprocess\nroot=Path('/home/ubuntu/spacepdhcg-joint-mesh-v666')\n"
code+="assert not (root/'archive-v670.py').exists()\n(root/'archive-v670.py').write_text("+repr(archive)+")\n(root/'archive-worker-v670.py').write_text("+repr(worker)+")\n"
code+="with (root/'archive-worker-v670.log').open('x') as log:child=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'archive-worker-v670.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid)))\n"
(p/'launch_mesh_archive_lambda_v670.py').write_text(code)
