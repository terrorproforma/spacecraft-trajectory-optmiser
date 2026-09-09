from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('results/lambda/2026-09-09/gpu-collect-composition-v807')
remote="from pathlib import Path\nimport json,tarfile,hashlib\np=Path.home()/'spacepdhcg-collect-composition-v807'\nr=json.loads((p/'report.json').read_text());assert r['complete'] and r['success'] and r['qualified']\nfiles=[x for x in p.rglob('*') if x.is_file() and 'official' not in x.relative_to(p).parts and x.suffix!='.gz']\nwith tarfile.open(p/'evidence.tar.gz','w:gz') as tar:\n for x in files:tar.add(x,arcname=str(x.relative_to(p)),recursive=False)\nprint(json.dumps(dict(sha256=hashlib.sha256((p/'evidence.tar.gz').read_bytes()).hexdigest(),report=r)))"
r=subprocess.run(['ssh','-i','/tmp/traj-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=remote,text=True,capture_output=True,check=True,timeout=45)
receipt=json.loads(r.stdout);(root/'h100-receipt.json').write_text(json.dumps(receipt,indent=2))
subprocess.run(['scp','-q','-i','/tmp/traj-key.pem','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-collect-composition-v807/evidence.tar.gz',str(root/'h100.tar.gz')],check=True,timeout=55)
assert hashlib.sha256((root/'h100.tar.gz').read_bytes()).hexdigest()==receipt['sha256']
import tarfile
with tarfile.open(root/'h100.tar.gz') as tar:
    for m in tar.getmembers():
        assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        if m.name=='Result.txt':assert tar.extractfile(m).read()==(root/'Result.txt').read_bytes();continue
        p=root/'h100'/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read())
local=Path.home()/'spacepdhcg-collect-composition-v807';r=json.loads((local/'report.json').read_text());assert r['complete'] and r['success'] and r['qualified']
assert (local/'Result.txt').read_bytes()==(root/'Result.txt').read_bytes()
for n in ('report.json','run.py','launch.py','official.stdout','official.stderr','worker.log'):
    (root/'local').mkdir(exist_ok=True);shutil.copyfile(local/n,root/'local'/n)
(root/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n')
(root/'reproduce').mkdir()
for n in ('compose_collect_v807.py','check_collect_v807.py','resume_collect_v807.py','retrieve_collect_v807.py'):shutil.copyfile(Path('build/performance')/n,root/'reproduce'/n)
print(json.dumps({side:json.loads((root/side/'report.json').read_text()) for side in ('local','h100')},indent=2))
