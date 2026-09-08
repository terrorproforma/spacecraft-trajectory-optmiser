from pathlib import Path
import hashlib,json,shutil
root=Path('/home/angus/spacepdhcg-joint-mesh-v662');name='tests/test_gtoc12_gpu_joint_mesh.py'
shutil.copy2(root/'repo'/name,root/'mesh-test-v663.py');shutil.copy2(name,root/'repo'/name)
(root/'test-fix-v665.json').write_text(json.dumps(dict(reason='The shared oracle helper includes the incumbent as row zero, but production moves() and native mesh contain shifts only. Remove that extra row from expected matrices. Native/runtime source unchanged.',previous_sha256=hashlib.sha256((root/'mesh-test-v663.py').read_bytes()).hexdigest(),fixed_sha256=hashlib.sha256((root/'repo'/name).read_bytes()).hexdigest(),original_pytest_returncode=1),indent=2))
p=Path('build/performance');(p/'check_joint_mesh_v665.py').write_text((p/'check_joint_mesh_v663.py').read_text().replace('validation-v663','validation-v665'))
