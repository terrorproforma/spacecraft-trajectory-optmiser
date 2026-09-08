from pathlib import Path
import shutil
p=Path('build/performance')
source=(p/'check_early_graph_v468.py').read_text()
source=source.replace('early-graph-v468','workspace-pool-v474')
source=source.replace("frozen=Path('/home/angus/build-spacepdhcg-early-graph-v465/final');core=frozen/'libspacepdhcg_cuda.so'", "frozen=Path('/home/angus/build-spacepdhcg-workspace-pool-v474/final');frozen.mkdir(parents=True,exist_ok=False);core=frozen/'libspacepdhcg_cuda.so';shutil.copy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',core)")
source=source.replace("sources=['cpp/cuda/src/native_qoco_adapter.cpp'", "sources=['cpp/cuda/src/gtoc12_qoco.cu','cpp/cuda/src/gtoc12_conic.cu','cpp/cuda/src/gtoc12_discretisation.cu','cpp/cuda/internal/gtoc12_workspace_reuse.h','cpp/cuda/internal/native_qoco_adapter.h','tests/test_gtoc12_gpu_workspace_pool.py','cpp/cuda/src/native_qoco_adapter.cpp'")
source=source.replace("'tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py'", "'tests/test_gtoc12_gpu_workspace_pool.py','tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py'")
(p/'check_workspace_pool_v474.py').write_text(source)
