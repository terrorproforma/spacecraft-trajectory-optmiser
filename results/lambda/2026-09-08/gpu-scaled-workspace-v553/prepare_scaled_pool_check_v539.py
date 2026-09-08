from pathlib import Path
p=Path('build/performance')
s=(p/'check_retained_replay_v512.py').read_text().replace('retained-replay-v512','scaled-pool-v539').replace('build-qoco-soc-step-v358','build-qoco-scaled-pool-v540')
s=s.replace("'tests/test_gtoc12_gpu_retained_replay.py','tests/test_gtoc12_gpu_workspace_pool.py'", "'tests/test_gtoc12_gpu_scaled_workspace_pool.py','tests/test_gtoc12_gpu_retained_replay.py','tests/test_gtoc12_gpu_workspace_pool.py'")
s=s.replace("sources=['cpp/cuda/src/native_qoco_adapter.cpp'", "sources=['tests/test_gtoc12_gpu_scaled_workspace_pool.py','scripts/gpu/prepare_qoco_preserve_objective.py','cpp/cuda/src/native_qoco_adapter.cpp'")
s=s.replace("env['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY']='1'", "env['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY']='1'\nenv['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1'\nenv['SPACEPDHCG_GTOC12_OLD_QOCO_LIBRARY']='/home/angus/build-qoco-preserve-objective-v534/final/libqoco.so'")
s=s.replace('child.wait(timeout=300)','child.wait(timeout=600)')
(p/'check_scaled_pool_v539.py').write_text(s)
