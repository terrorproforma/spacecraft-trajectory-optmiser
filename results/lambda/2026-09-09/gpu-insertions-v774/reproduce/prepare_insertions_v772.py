from pathlib import Path
p=Path('build/performance/prepare_insertions_v770.py').read_text().replace('v770','v772')
p=p.replace("for name in ['src/spacepdhcg/gtoc12/gpu_joint_insertions.py','tests/test_gtoc12_gpu_joint_insertions.py']:","for name in ['src/spacepdhcg/gtoc12/gpu_joint.py','src/spacepdhcg/gtoc12/jointopt.py','src/spacepdhcg/gtoc12/gpu_joint_insertions.py','tests/test_gtoc12_gpu_joint_insertions.py']:")
exec(compile(p,__file__,'exec'))
