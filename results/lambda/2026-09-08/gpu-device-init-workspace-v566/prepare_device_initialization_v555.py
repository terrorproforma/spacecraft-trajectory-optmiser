from pathlib import Path
p=Path('build/performance')
s=(p/'check_scaled_pool_v539.py').read_text().replace('scaled-pool-v539','device-init-v555')
s=s.replace("'tests/test_gtoc12_gpu_scaled_workspace_pool.py'", "'tests/test_gtoc12_gpu_device_initialization.py'")
s=s.replace("env['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1'", "env['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1'\nenv['SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION']='1'")
(p/'check_device_initialization_v555.py').write_text(s)
