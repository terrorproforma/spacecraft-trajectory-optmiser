from pathlib import Path
p=Path('build/performance')
s=(p/'validate_device_initialization_v558.py').read_text().replace('device-init-v558','device-init-v562').replace('validate_device_initialization_v558.py','validate_device_initialization_v562.py')
s=s.replace("files=['tests/test_gtoc12_gpu_qoco.py'", "files=['tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_qoco.py'")
a=s.index(" tests=['tests/'+name");b=s.index('\n for name in tests:',a)
s=s[:a]+" tests=['tests/'+name for name in ['test_gtoc12_gpu_scvx.py','test_gtoc12_gpu_cli.py','test_gtoc12_run_final_verification.py']]"+s[b:]
(p/'validate_device_initialization_v562.py').write_text(s)
s=(p/'prepare_device_initialization_v559.py').read_text().replace('v558','v562').replace('v559','v563')
s=s.replace("'tests/test_gtoc12_gpu_qoco.py':Path('tests/test_gtoc12_gpu_qoco.py').read_text()", "'tests/test_gtoc12_gpu_qoco.py':Path('tests/test_gtoc12_gpu_qoco.py').read_text(),'tests/test_gtoc12_gpu_scvx.py':Path('tests/test_gtoc12_gpu_scvx.py').read_text()")
(p/'prepare_device_initialization_v563.py').write_text(s)
