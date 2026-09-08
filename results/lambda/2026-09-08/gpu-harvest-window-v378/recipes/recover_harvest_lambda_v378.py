from pathlib import Path
import os,subprocess,json
root=Path('/home/ubuntu/spacepdhcg-harvest-window-v378');repo=root/'repo'
old=json.loads((root/'report.json').read_text())
assert old.get('error') and not old['complete']
assert not Path('/proc/165920').exists() and not Path('/proc/165921').exists()
# CMake records the source revision. Give this isolated, fully copied source
# snapshot its own revision; do not reuse another checkout's Git metadata.
for cmd in [['git','init','-q'],['git','add','-A'],['git','-c','user.name=Dirtman','-c','user.email=130276587+terrorproforma@users.noreply.github.com','commit','-q','-m','Frozen H100 harvest-window validation source']]:
 subprocess.run(cmd,cwd=repo,check=True)
old['frozen_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
(root/'report.json').write_text(json.dumps(old,indent=2))
s=(root/'run.py').read_text().replace("report=dict(pid=os.getpid(),complete=False,stages=[],campaigns=[])","report=json.loads((root/'report.json').read_text());report.update(pid=os.getpid(),complete=False);report.pop('error',None)")
a=s.index(" shutil.copytree(");b=s.index(" cmake=",a);s=s[:a]+s[b:]
s=s.replace("run('configure',","run('configure_retry',")
compile(s,'run-retry.py','exec');(root/'run-retry.py').write_text(s)
with (root/'runner-retry.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run-retry.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,source_commit=old['frozen_source_commit'])))
