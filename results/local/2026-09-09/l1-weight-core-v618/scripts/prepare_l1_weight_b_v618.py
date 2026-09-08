"""Retain failed configure evidence; prepare b with CMake-only source correction."""
from pathlib import Path
import shutil
live=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
attempt=Path('/home/angus/spacepdhcg-l1-weight-v618a')
destination=live/'build/performance/l1-weight-v618a';destination.mkdir(exist_ok=False)
for p in attempt.iterdir():
    if p.is_file():shutil.copy2(p,destination/p.name)
shutil.copytree(attempt/'fixtures',destination/'fixtures')
script=(live/'build/performance/build_l1_weight_v618a.py').read_text().replace('spacepdhcg-l1-weight-v618a','spacepdhcg-l1-weight-v618b').replace("l1-weight-v618a'","l1-weight-v618b'")
script=script.replace("'git_operations':0","'git_mutations':0,'cmake_uses_readonly_pinned_git_checks':True")
begin=script.index("commands=subprocess.check_output(['ninja'")
end=script.index("manifest['library_sha256']=",begin)
reuse="""reuse=Path('/home/angus/spacepdhcg-l1-weight-v618a')
previous=json.loads((reuse/'manifest.json').read_text())
assert all(s['returncode']==0 for s in previous['stages'] if s['name']!='cmake-configure')
for name in owned:
    if name!='cpp/cuda/CMakeLists.txt':assert digest(repo/name)==previous['source_sha256'][name],name
assert digest(reuse/'build/cuda/libspacepdhcg_cuda.so')==previous['library_sha256']
shutil.copy2(reuse/'build/cuda/libspacepdhcg_cuda.so',build/'cuda/libspacepdhcg_cuda.so')
manifest['reused_candidate_core']={'path':str(reuse/'build/cuda/libspacepdhcg_cuda.so'),
    'manifest_sha256':digest(reuse/'manifest.json'),'source_tree_sha256':previous['source_tree_sha256'],
    'reason':'only CMake frozen-archive provenance changed; all core/test/importer sources identical'}
"""
script=script[:begin]+reuse+script[end:]
needle="'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER='+nvcc,"
script=script.replace(needle,"'-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_FROZEN_SOURCE_TREE_SHA256='+manifest['source_tree_sha256'],\n    '-DSPACEPDHCG_FROZEN_SOURCE_BASE_COMMIT='+manifest['base_commit'],'-DCMAKE_CUDA_COMPILER='+nvcc,")
target=live/'build/performance/build_l1_weight_v618b.py'
with target.open('x') as f:f.write(script)
print(target)
