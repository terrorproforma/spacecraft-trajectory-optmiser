from pathlib import Path
import json,subprocess
home=Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus')
root=home/'spacepdhcg-certificate-v709/followup-v710'
assert 'SyntaxError: unterminated string literal' in (root/'worker.log').read_text()
assert not (root/'report.json').exists()
worker=root/'worker-v711.py'
worker.write_text((root/'worker.py').read_text().replace(".split('report=dict(')",".split('\\nreport=dict(')"))
with (root/'worker-v711.log').open('x') as log:
    child=subprocess.Popen(['python3',str(worker)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(root=str(root),pid=child.pid)))
