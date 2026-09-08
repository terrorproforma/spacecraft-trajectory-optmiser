from pathlib import Path
p=Path('build/performance')
s=(p/'check_device_initialization_v564.py').read_text().replace('device-init-v564','bound-types-v567')
s=s.replace("env.pop('SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION',None)","env['SPACEPDHCG_TEST_QOCO_DEVICE_BOUND_TYPES']='1'")
s=s.replace("sources=['tests/test_gtoc12_gpu_qoco.py'", "sources=['tests/test_gtoc12_gpu_bound_types.py','cpp/cuda/src/native_qoco_gpu.cu','cpp/cuda/internal/native_qoco_gpu.h','cpp/cuda/tests/qoco_gpu_bound_types_test.cu','tests/test_gtoc12_gpu_qoco.py'")
s=s.replace("boot,'tests/test_gtoc12_gpu_qoco.py'", "boot,'tests/test_gtoc12_gpu_bound_types.py','tests/test_gtoc12_gpu_qoco.py'")
start=s.index("binary=");end=s.index("py=",start)
s=s[:start]+s[end:]
start=s.index("r=dict(")
s=s[:start]+"probe=str(root.resolve()/'bound-types-test')\njobs.insert(0,('compile-bound-types',['/usr/local/cuda-12.8/bin/nvcc','-std=c++17','--fmad=false','-arch=sm_120','cpp/cuda/tests/qoco_gpu_bound_types_test.cu','-L'+str(core.parent),'-lspacepdhcg_cuda','-o',probe]))\njobs.insert(1,('bound-types',[probe]))\n"+s[start:]
(p/'check_bound_types_v567.py').write_text(s)
