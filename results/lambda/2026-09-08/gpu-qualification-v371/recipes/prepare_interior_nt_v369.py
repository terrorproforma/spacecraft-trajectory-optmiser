from pathlib import Path
s=Path('build/performance/build_final_step_v358.py').read_text().replace('build-qoco-soc-step-v358','build-qoco-interior-nt-v369').replace("shutil.copytree('/home/angus/build-qoco-nonfinite-ir-v336/source'","shutil.copytree('/home/angus/build-qoco-interior-step-v367/source'")
a=s.index('from prepare_qoco_soc_step import prepare');b=s.index('cmake=',a)
patch=r'''
header=Path('build/performance/nt-normalization-v347/arithmetic.cuh').read_text()
first=header.index('namespace qoco_cone_arithmetic {');second=header.index('namespace qoco_cone_arithmetic {',first+1)
(source/'src/qoco_nt_precision.cuh').write_text('#pragma once\n'+header[second:])
p=source/'src/cone.cu';text=p.read_text()
a=text.index('__global__ void compute_nt_scaling_kernel');b=text.index('__global__ void nt_multiply_kernel',a)
text=text[:a]+'#include "qoco_soc_step.cuh"\n#include "qoco_nt_precision.cuh"\n'+Path('build/performance/nt-normalization-v347/new_kernel.cuh').read_text()+text[b:]
p.write_text(text)
'''
Path('build/performance/build_interior_nt_v369.py').write_text(s[:a]+patch+s[b:])
r=Path('build/performance/replay_interior_step_v367.py').read_text().replace('interior-step-replay-v367','interior-nt-replay-v369').replace('build-qoco-interior-step-v367','build-qoco-interior-nt-v369').replace('version==367','version==369').replace(',367)',',369)').replace("'interior_step","'interior_nt")
Path('build/performance/replay_interior_nt_v369.py').write_text(r)
