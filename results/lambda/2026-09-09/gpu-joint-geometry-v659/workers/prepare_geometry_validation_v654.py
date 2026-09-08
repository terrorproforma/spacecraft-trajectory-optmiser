from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/home/angus/spacepdhcg-joint-geometry-v651')
name='tests/test_gtoc12_gpu_joint_geometry.py'
original=root/'repo'/name
shutil.copy2(original,root/'geometry-test-v652.py')
shutil.copy2(name,original)
(root/'test-update-v654.json').write_text(json.dumps({'original_sha256':hashlib.sha256((root/'geometry-test-v652.py').read_bytes()).hexdigest(),'final_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'native_and_wrapper_unchanged':True},indent=2))
runner=Path('build/performance/check_joint_geometry_v652.py').read_text().replace("'validation-v652'","'validation-v654'")
(root/'validation-runner-v654.py').write_text(runner)
with (root/'validation-runner-v654.log').open('x') as log:
 child=subprocess.Popen(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',str(root/'validation-runner-v654.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':child.pid}))
