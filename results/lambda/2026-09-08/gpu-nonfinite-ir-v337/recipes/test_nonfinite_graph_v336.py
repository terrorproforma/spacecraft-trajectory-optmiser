from pathlib import Path
import subprocess,os,json
source=Path('/home/angus/build-qoco-nonfinite-ir-v336/source');lib=source.parent/'final';root=Path('build/performance/nonfinite-ir-v336')
cmd=['/usr/local/cuda-12.8/bin/nvcc','-O2','-std=c++17','-arch=sm_120','--default-stream','per-thread','cpp/cuda/tests/qoco_ir_nonfinite_graph_probe.cu','-Icpp/cuda/patches',*['-I'+str(source/p) for p in ['algebra/cuda','include','lib/qdldl/include']],'-I/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/include','-ldl','-o',str(lib/'qoco_ir_nonfinite_graph_probe')]
r=subprocess.run(cmd,capture_output=True,text=True);print(r.stdout,r.stderr);r.check_returncode()
r=subprocess.run([str(lib/'qoco_ir_nonfinite_graph_probe')],capture_output=True,text=True);(root/'actual-graph.log').write_text(r.stdout+r.stderr);print(r.stdout,r.stderr);r.check_returncode()
