from pathlib import Path
p=Path('build/performance');s=(p/'check_device_initialization_v555.py').read_text().replace('v555','v564')
s=s.replace("env['SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION']='1'", "env.pop('SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION',None)")
s=s.replace("sources=['tests/test_gtoc12_gpu_device_initialization.py'", "sources=['tests/test_gtoc12_gpu_qoco.py','tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_device_initialization.py'")
s=s.replace("boot,'tests/test_gtoc12_gpu_device_initialization.py'", "boot,'tests/test_gtoc12_gpu_qoco.py','tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_device_initialization.py'")
(p/'check_device_initialization_v564.py').write_text(s)
