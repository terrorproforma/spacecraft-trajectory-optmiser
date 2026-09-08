from pathlib import Path
import hashlib,json,shutil,subprocess
p=Path('build/performance');root=Path('/home/angus/spacepdhcg-grid-v788');root.mkdir()
old=root.with_name('spacepdhcg-grid-v786')
shutil.copytree(old/'repo',root/'repo',ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache','.ruff_cache'))
for name in ('incumbents','final'):shutil.copytree(old/name,root/name)
overlays=['src/spacepdhcg/gtoc12/jointopt.py','src/spacepdhcg/gtoc12/gpu_joint_insertions.py','tests/test_gtoc12_gpu_insertion_grid.py']
for name in overlays:shutil.copyfile(name,root/'repo'/name)
m=json.loads((old/'source-manifest.json').read_text());m['core_source']='v786: unchanged C++';m['files']={f.relative_to(root/'repo').as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'repo').rglob('*') if f.is_file()}
(root/'source-manifest.json').write_text(json.dumps(m,indent=2))
worker=(old/'worker.py').read_text();a=worker.index("    run('configure'");b=worker.index("    report['core_sha256']",a);worker=worker[:a]+worker[b:]
(root/'worker.py').write_text(worker)
with (root/'worker.log').open('x') as log:print('local',subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
send=(p/'send_grid_v786.py').read_text().replace('spacepdhcg-grid-v786','spacepdhcg-grid-v788').replace('grid-source-v786','grid-source-v788').replace('launch_grid_h100_v786','launch_grid_h100_v788')
send=send.replace('import hashlib,json,subprocess,tarfile','import hashlib,json,shutil,subprocess,tarfile')
send=send.replace("with (root/'worker.log').open('x') as log:", "shutil.copytree(root.with_name('spacepdhcg-grid-v786')/'final',root/'final')\nwith (root/'worker.log').open('x') as log:")
(p/'send_grid_v788.py').write_text(send)
(p/'status_grid_v788.py').write_text((p/'status_grid_v786.py').read_text().replace('v786','v788'))
