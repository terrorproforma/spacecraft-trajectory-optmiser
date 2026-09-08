from pathlib import Path
base=Path('build/performance/run_solver_phase_v461.py').read_text().replace('solver-phase-v461','early-graph-v463').replace('solver_phase461','early_graph463').replace('build-spacepdhcg-solver-phase-v458','build-spacepdhcg-early-graph-v463')
base=base.replace("r=dict(source_sha256=", "from shutil import copy2\nPath(core).parent.mkdir(parents=True,exist_ok=False)\ncopy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',core)\ncopy2('cpp/cuda/src/native_qoco_adapter.cpp',root/'native_qoco_adapter.cpp')\ncopy2('cpp/cuda/src/gtoc12_scvx.cu',root/'gtoc12_scvx.cu')\nenv['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='1'\nr=dict(source_sha256=")
base=base.replace("'cpp/cuda/src/gtoc12_scvx.cu','build/performance/solver_phase_details.py'", "'cpp/cuda/src/gtoc12_scvx.cu','cpp/cuda/src/native_qoco_adapter.cpp','build/performance/solver_phase_details.py'")
Path('build/performance/run_early_graph_v463.py').write_text(base)
Path('build/performance/run_early_graph_v465.py').write_text(base.replace('early-graph-v463','early-graph-v465').replace('early_graph463','early_graph465'))
base=Path('build/performance/check_stationary_v411.py').read_text().replace('stationary-v411','early-graph-v464').replace('stationary-probe-v411','early-graph-probe-v464')
start=base.index('frozen=');end=base.index('sources=',start)
base=base[:start]+"frozen=Path('/home/angus/build-spacepdhcg-early-graph-v465/final');core=frozen/'libspacepdhcg_cuda.so'\n"+base[end:]
base=base.replace("sources=[", "sources=['cpp/cuda/src/native_qoco_adapter.cpp','tests/test_gtoc12_gpu_scvx.py',")
base=base.replace('env.update(PYTHONPATH=', "env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='1'\nenv.update(PYTHONPATH=")
Path('build/performance/check_early_graph_v464.py').write_text(base)
