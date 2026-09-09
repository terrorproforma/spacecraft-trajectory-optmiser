from pathlib import Path
import hashlib,json,shutil,subprocess
repo=Path(__file__).resolve().parents[2]
out=repo/'results/lambda/2026-09-09/gpu-native-collection-plan-v859';out.mkdir(exist_ok=True)
script=(repo/'build/performance/package_collect_plan_v859.py').read_text()
ssh=['ssh','-i','/home/angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229']
r=subprocess.run(ssh+['python3 -'],input=script,text=True,capture_output=True,timeout=55,check=True);print('H100',r.stdout,r.stderr)
subprocess.run(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',str(repo/'build/performance/package_collect_plan_v859.py')],check=True,timeout=55)
remote='/home/ubuntu/spacepdhcg-native-collect-plan-evidence-v859/'
local=Path.home()/'spacepdhcg-native-collect-plan-evidence-v859'
for name in ('h100.tar.gz','h100-manifest.json'):
    assert not (out/name).exists()
    subprocess.run(['scp','-q','-i','/home/angus/.ssh/spacepdhcg-expansion-key.pem','ubuntu@192.222.55.229:'+remote+name,str(out/name)],check=True,timeout=100)
for name in ('local.tar.gz','local-manifest.json'):
    assert not (out/name).exists();shutil.copyfile(local/name,out/name)
for side in ('local','h100'):
    manifest=json.loads((out/(side+'-manifest.json')).read_text())
    assert hashlib.sha256((out/(side+'.tar.gz')).read_bytes()).hexdigest()==manifest['archive_sha256']
    print(side,manifest['archive_bytes'],len(manifest['files']))
