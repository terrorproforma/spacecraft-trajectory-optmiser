from pathlib import Path
p=Path('build/performance')
s=(p/'replay_scaled_pool.py').read_text()
s=s.replace("os.environ['SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL']=mode", "os.environ['SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL']='0'\nos.environ['SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION']=mode")
s=s.replace("os.environ['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1'", "os.environ['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='0'").replace('qoco_ruiz_iterations=2','qoco_ruiz_iterations=0')
(p/'replay_device_initialization.py').write_text(s)
s=(p/'validate_scaled_pool_v542.py').read_text().replace('scaled-pool-v542','device-init-v556').replace('build-spacepdhcg-scaled-pool-v539','build-spacepdhcg-device-init-v555')
s=s.replace('replay_scaled_pool.py','replay_device_initialization.py').replace('validate_scaled_pool_v542.py','validate_device_initialization_v556.py')
s=s.replace("tests=['tests/'+name for name in [", "tests=['tests/'+name for name in ['test_gtoc12_gpu_device_initialization.py','test_gtoc12_gpu_qoco.py',")
s=s.replace("files=['cpp/cuda/src/gtoc12_qoco.cu'", "files=['tests/test_gtoc12_gpu_device_initialization.py','cpp/cuda/src/gtoc12_qoco.cu'")
s=s.replace("report=dict(pid=", "env['SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION']='1'\nreport=dict(pid=")
a=s.index("  selection=");b=s.index('  replay=',a);s=s[:a]+s[b:]
(p/'validate_device_initialization_v556.py').write_text(s)
