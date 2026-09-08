from pathlib import Path
s=Path('build/performance/build_nt_step_v349.py').read_text().replace('build-qoco-nt-step-v349','build-qoco-step-only-v352')
a=s.index("a=text.index('__global__ void compute_nt_scaling_kernel')");b=s.index("a=text.index('__device__ QOCOFloat soc_step_length_dev(')",a)
s=s[:a]+'text=\'#include "qoco_cone_arithmetic.cuh"\\n\'+text\n'+s[b:]
Path('build/performance/build_step_only_v352.py').write_text(s)
r=Path('build/performance/replay_nt_step_v349.py').read_text().replace('nt-step','step-only').replace('349','352').replace('nt_step','step_only')
Path('build/performance/replay_step_only_v352.py').write_text(r)
