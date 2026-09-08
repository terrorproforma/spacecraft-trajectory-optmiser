from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/home/angus/spacepdhcg-insertions-v770');root.mkdir()
old=root.with_name('spacepdhcg-insertions-v768')
shutil.copytree(old/'repo',root/'repo',ignore=shutil.ignore_patterns('.git'))
shutil.copytree(old/'incumbents',root/'incumbents');shutil.copytree(old/'final',root/'final')
for name in ['src/spacepdhcg/gtoc12/gpu_joint_insertions.py','tests/test_gtoc12_gpu_joint_insertions.py']:
    shutil.copyfile(name,root/'repo'/name)
manifest=json.loads((old/'source-manifest.json').read_text());manifest['files']={p.relative_to(root/'repo').as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'repo').rglob('*') if p.is_file() and '__pycache__' not in p.parts};manifest['core_source']='v768: unchanged C++'
(root/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
worker=Path('build/performance/worker_insertions_v768.py').read_text()
start=worker.index("    run('configure'");end=worker.index("    report['core_sha256']",start)
worker=worker[:start]+worker[end:]
worker=worker.replace("run('pytest',[py,'-c',boot,'-q','tests/test_gtoc12_gpu_joint_insertions.py'])", "run('pytest',[py,'-c',boot,'-q',*sorted(str(p.relative_to(repo)) for p in (repo/'tests').glob('test_gtoc12_gpu_joint*.py')),'tests/test_gtoc12_jointopt.py'])")
(root/'worker.py').write_text(worker)
with (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
send=Path('build/performance/send_insertions_v768.py').read_text().replace('v768','v770')
send=send.replace("for name in ('worker.py','source-manifest.json','incumbents'):","for name in ('worker.py','source-manifest.json','incumbents'):")
send=send.replace("import hashlib,json,subprocess,tarfile", "import hashlib,json,shutil,subprocess,tarfile")
send=send.replace("with (root/'worker.log').open('x') as log:","shutil.copytree(root.with_name('spacepdhcg-insertions-v768')/'final',root/'final')\nwith (root/'worker.log').open('x') as log:")
Path('build/performance/send_insertions_v770.py').write_text(send)
Path('build/performance/status_insertions_v770.py').write_text(Path('build/performance/status_insertions_v768.py').read_text().replace('v768','v770'))
