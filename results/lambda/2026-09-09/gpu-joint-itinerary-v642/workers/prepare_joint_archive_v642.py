from pathlib import Path
p=Path('build/performance')
code="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-joint-selection-v632')\n"
code+="(root/'archive.py').write_text("+repr((p/'archive_joint_v642.py').read_text())+")\n"
waiter="from pathlib import Path\nimport json,subprocess,time\nroot=Path('/home/ubuntu/spacepdhcg-joint-selection-v632');ready=root/'compatibility-validation-v640/report.json'\nwhile not ready.exists():time.sleep(10)\nassert json.loads(ready.read_text())['returncode']==0\nassert json.loads((root/'campaign-v636/report.json').read_text())['success']\nraise SystemExit(subprocess.call(['python3',str(root/'archive.py'),str(root),'/home/ubuntu/joint-selection-v642-h100.tar.gz','/home/ubuntu/spacepdhcg-scaled-pool-v545/final/libqoco.so']))\n"
code+="(root/'wait-archive.py').write_text("+repr(waiter)+")\n"
code+="with (root/'archive-runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'wait-archive.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid)))\n"
(p/'launch_joint_archive_v642.py').write_text(code)
