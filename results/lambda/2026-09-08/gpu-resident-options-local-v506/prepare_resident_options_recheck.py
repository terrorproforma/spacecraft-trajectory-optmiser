from pathlib import Path
import ast,json,hashlib,tarfile,os,subprocess
p=Path('build/performance')
extra=['src/spacepdhcg/gtoc12/verifier.py','tests/test_gtoc12_verifier_knots.py','tests/fixtures/gtoc12_lagrange_replay.json','tests/test_gtoc12_verifier.py','tests/test_gtoc12_gpu_verifier.py']
run=(p/'check_resident_options_v498.py').read_text().replace('resident-options-v498','resident-options-v506')
a=run.index('sources=');b=run.index('\nfor name in sources:',a);sources=list(dict.fromkeys(ast.literal_eval(run[a+8:b])+extra));run=run[:a]+'sources='+repr(sources)+run[b:]
run=run.replace("boot,'tests/test_gtoc12_gpu_resident_options.py'", "boot,'tests/test_gtoc12_verifier_knots.py','tests/test_gtoc12_verifier.py','tests/test_gtoc12_gpu_verifier.py','tests/test_gtoc12_gpu_resident_options.py'")
(p/'check_resident_options_v506.py').write_text(run)
remote=(p/'run_resident_options_v500.py').read_text().replace('resident-options-v500','resident-options-v507')
remote=remote.replace("tests=['tests/test_gtoc12_gpu_resident_options.py'", "tests=['tests/test_gtoc12_verifier_knots.py','tests/test_gtoc12_verifier.py','tests/test_gtoc12_gpu_verifier.py','tests/test_gtoc12_gpu_resident_options.py'")
(p/'run_resident_options_v507.py').write_text(remote)
names=list(dict.fromkeys(list(json.loads((p/'resident-options-source-sha256-v500.json').read_text()))+sources))
mp=p/'resident-options-source-sha256-v507.json';mp.write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in names},indent=2))
archive=Path('/tmp/resident-options-v507.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for f in names:t.add(f,arcname=f)
 t.add(mp,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=180)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-resident-options-v507');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(remote)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_resident_options_v507.py').write_text(launch)
(p/'status_resident_options_v507.py').write_text((p/'status_resident_options_v500.py').read_text().replace('resident-options-v500','resident-options-v507'))
