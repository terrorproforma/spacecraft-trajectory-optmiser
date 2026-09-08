
from pathlib import Path
import fcntl, hashlib, json, os, pstats, subprocess, time, traceback
home = Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus')
root = Path(__file__).resolve().parent
old = home / 'spacepdhcg-search-campaign-v703/worker-v706.py'
namespace = {'__file__': str(old)}
exec(old.read_text().split('report=dict(')[0], namespace)
source, environment, python = (namespace[k] for k in ('source', 'environment', 'python'))
environment['SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SEARCH'] = '1'
command = [python, '-m', 'cProfile', '-o', str(root/'campaign.pstats'), str(source/'build/performance/orphan-recovery-v595/run.py'), '--repo', str(source), '--output', str(root/'campaign'), '--wall-seconds', '1800', '--joint-seconds', '15', '--max-certifications', '4', '--lock', str(root/'child.lock')]
report = dict(complete=False, pid=os.getpid(), command=command, core_sha256=hashlib.sha256(namespace['library'].read_bytes()).hexdigest())
def save():
    temporary = root/'profile.tmp'
    temporary.write_text(json.dumps(report, indent=2)); temporary.replace(root/'profile.json')
save()
try:
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started=time.perf_counter()
        with (root/'campaign.log').open('x') as log:
            child=subprocess.Popen(command,cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT)
            report['child_pid']=child.pid; save()
            report['returncode']=child.wait()
        report['process_seconds']=time.perf_counter()-started
    if (root/'campaign.pstats').exists():
        with (root/'profile.txt').open('w') as stream:
            stats=pstats.Stats(str(root/'campaign.pstats'),stream=stream)
            stats.strip_dirs().sort_stats('cumulative').print_stats(70)
            stats.sort_stats('tottime').print_stats(50)
    report['success']=report['returncode']==0
except BaseException:
    report['error']=traceback.format_exc()
report['complete']=True; save()
