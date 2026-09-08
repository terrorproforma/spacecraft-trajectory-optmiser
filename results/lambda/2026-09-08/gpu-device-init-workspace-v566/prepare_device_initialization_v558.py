from pathlib import Path
p=Path('build/performance')
s=(p/'validate_device_initialization_v556.py').read_text().replace('device-init-v556','device-init-v558').replace('validate_device_initialization_v556.py','validate_device_initialization_v558.py')
s=s.replace("files=['tests/test_gtoc12_gpu_device_initialization.py'", "files=['tests/test_gtoc12_gpu_qoco.py','tests/test_gtoc12_gpu_device_initialization.py'")
(p/'validate_device_initialization_v558.py').write_text(s)
