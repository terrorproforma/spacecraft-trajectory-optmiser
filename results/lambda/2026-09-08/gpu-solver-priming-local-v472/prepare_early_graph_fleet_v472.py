from pathlib import Path
base=Path('build/performance/run_fused_tables_fleet_v455.py').read_text().replace('fused-tables-fleet-v455','early-graph-fleet-v472').replace('fused_fleet455','early_graph_fleet472').replace('build-spacepdhcg-fused-tables-v454','build-spacepdhcg-early-graph-v465')
base=base.replace('env.update(PYTHONPATH=', "env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='1'\nenv.update(PYTHONPATH=")
base=base.replace("'cpp/cuda/src/gtoc12_collect_dp.cu',", "'cpp/cuda/src/gtoc12_collect_dp.cu','cpp/cuda/src/gtoc12_scvx.cu','cpp/cuda/src/native_qoco_adapter.cpp',")
Path('build/performance/run_early_graph_fleet_v472.py').write_text(base)
