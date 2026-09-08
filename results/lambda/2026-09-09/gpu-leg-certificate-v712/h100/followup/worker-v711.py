from pathlib import Path
import fcntl,json,subprocess,time,traceback
root=Path(__file__).resolve().parent;base=root.parent
home=Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus')
namespace={'__file__':str(base/'worker.py')}
exec((base/'worker.py').read_text().split('\nreport=dict(')[0],namespace)
py=namespace['py'];env=dict(namespace['env'],PYTHONPATH=str(root/'repo/src'))
boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
report=dict(pid=__import__('os').getpid(),complete=False,success=False,stages=[])
def save():
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(root/'report.json')
save()
try:
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert json.loads((base/'report.json').read_text())['success']
        commands=[('pytest',[py,'-c',boot,'-q','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_certificate_backend.py'])]
        launch="import runpy,sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
        commands.append(('benchmark',[py,'-c',launch,str(root/'benchmark.py'),str(base),str(root/'benchmark.json')]))
        for name,command in commands:
            started=time.perf_counter()
            with (root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=root/'repo',env=env,stdout=log,stderr=subprocess.STDOUT)
                report.update(child_pid=child.pid,stage=name);save();code=child.wait()
            report['stages'].append(dict(name=name,code=code,seconds=time.perf_counter()-started,command=command));save()
            if code:raise RuntimeError((name,code))
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
