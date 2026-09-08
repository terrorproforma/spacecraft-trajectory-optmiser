from pathlib import Path
import os,subprocess,fcntl,json,time,hashlib
root=Path('/home/angus/build-qoco-nt-snapshot-v348');out=root/'snapshots';out.mkdir(exist_ok=False)
lib=root/'final/libqoco.so';binary=Path('build/performance/qp-regularization-v169/qoco_snapshot_replay').resolve();qp=Path('build/performance/qp-ir-v309/qp.txt').resolve()
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(LD_LIBRARY_PATH=str(lib.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64',SPACEPDHCG_DIAGNOSTIC_NT_DIR=str(out))
assert str(lib) in subprocess.check_output(['ldd',str(binary)],env=env,text=True)
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.perf_counter();r=subprocess.run([str(binary),str(qp),'1'],env=env,capture_output=True,text=True,timeout=120)
 (root/'replay.log').write_text(r.stdout);(root/'replay.err').write_text(r.stderr);r.check_returncode()
 record=[json.loads(s[10:]) for s in r.stdout.splitlines() if s.startswith('QP_REPLAY ')][0]
 print({k:v for k,v in record.items() if not isinstance(v,list)})
 print('snapshots',len(list(out.glob('*.bin'))),'seconds',time.perf_counter()-start)
