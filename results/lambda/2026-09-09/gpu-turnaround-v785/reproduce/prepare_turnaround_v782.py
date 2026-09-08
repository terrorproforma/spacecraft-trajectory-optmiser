from pathlib import Path
import hashlib,io,json,shutil,subprocess,tarfile
p=Path('build/performance');root=Path('/home/angus/spacepdhcg-turnaround-v782');root.mkdir()
repo=root/'repo';repo.mkdir();base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',base,'cpp','src','tests','scripts','benchmarks','pyproject.toml','third_party']))) as tar:tar.extractall(repo,filter='data')
owned=['src/spacepdhcg/gtoc12/jointopt.py','src/spacepdhcg/gtoc12/gpu_joint_insertions.py','src/spacepdhcg/gtoc12/gpu_joint_layouts.py','tests/test_gtoc12_gpu_joint_insertions.py','tests/test_gtoc12_gpu_joint_layouts.py','tests/test_gtoc12_insertion_pivot.py']
for name in owned:shutil.copyfile(name,repo/name)
old=root.with_name('spacepdhcg-layouts-v778')
shutil.copytree(old/'incumbents',root/'incumbents');shutil.copytree(old/'final',root/'final')
(root/'source-manifest.json').write_text(json.dumps(dict(base=base,overlays=owned,core_source='v776 unchanged C++',files={p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.rglob('*') if p.is_file()}),indent=2))
worker=(old/'worker.py').read_text().replace("'tests/test_gtoc12_jointopt.py'])", "'tests/test_gtoc12_jointopt.py','tests/test_gtoc12_insertion_pivot.py'])")
(root/'worker.py').write_text(worker)
with (root/'worker.log').open('x') as log:print('local',subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
send=(p/'send_layouts_v778.py').read_text().replace('spacepdhcg-layouts-v778','spacepdhcg-turnaround-v782').replace('layouts-source-v778','turnaround-source-v782').replace('launch_layouts_h100_v778','launch_turnaround_h100_v782')
(p/'send_turnaround_v782.py').write_text(send)
(p/'status_turnaround_v782.py').write_text((p/'status_layouts_v778.py').read_text().replace('spacepdhcg-layouts-v778','spacepdhcg-turnaround-v782'))
