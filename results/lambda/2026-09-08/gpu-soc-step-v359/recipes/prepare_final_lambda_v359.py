from pathlib import Path
s=Path('build/performance/run_step_lambda_v353.py').read_text().replace('spacepdhcg-step-v353','spacepdhcg-step-final-v359')
a=s.index(" patch=overlay/");b=s.index(' for node in ast.parse',a)
s=s[:a]+''' sys.path.insert(0,str(overlay/'scripts/gpu'))
 from prepare_qoco_soc_step import prepare
 report['preparation']=prepare(source);p=source/'src/cone.cu';save()
'''+s[b:]
s=s.replace("str(patch/'probe.cu')", "str(overlay/'cpp/cuda/tests/qoco_soc_step_probe.cu'),'-I'+str(overlay/'cpp/cuda/patches')")
s=s.replace("fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)","print('waiting for GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)")
a=s.index("  run('probe-live'");b=s.index("  for name,candidate",a)
s=s[:a]+'''  run('probe',[str(probe)])
  for tool in ['memcheck','initcheck','racecheck','synccheck']:
   run('probe-'+tool,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(probe)])
  repo=Path('/home/ubuntu/spacepdhcg-gpu-execution-v328/repo')
  env.update(PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_QOCO_LIBRARY=str(lib),SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/ubuntu/spacepdhcg-conic-retry-v314/core-build/cuda/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data')
  boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
  run('pytest',['/home/ubuntu/spacepdhcg/v1/.venv/bin/python','-c',boot,str(repo/'tests/test_gtoc12_gpu_scvx.py'),str(repo/'tests/test_gtoc12_gpu_cli.py'),'-q'])
'''+s[b:]
s=s.replace('[(354,True),(355,False),(356,False),(357,True)]','[(360,True)]')
compile(s,'run_final_lambda_v359','exec');Path('build/performance/run_final_lambda_v359.py').write_text(s)
d=Path('build/performance/dispatch_step_lambda_v353.py').read_text().replace('step-v353','step-final-v359')
a=d.index(" for d,names in ");b=d.index(" t.add('build/performance/qp-ir-v309/audit.py'",a)
d=d[:a]+''' for name in ['cpp/cuda/patches/qoco_soc_step.cuh','cpp/cuda/tests/qoco_soc_step_probe.cu','scripts/gpu/prepare_qoco_soc_step.py']:t.add(name,arcname=name)
'''+d[b:]
d=d.replace('run_step_lambda_v353.py','run_final_lambda_v359.py')
Path('build/performance/dispatch_final_lambda_v359.py').write_text(d)
p=Path('build/performance/poll_step_lambda_v353.py').read_text().replace('spacepdhcg-step-v353','spacepdhcg-step-final-v359')
Path('build/performance/poll_final_lambda_v359.py').write_text(p)
