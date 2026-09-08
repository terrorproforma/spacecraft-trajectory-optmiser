from pathlib import Path
import base64,subprocess
files={name:base64.b64encode(Path(name).read_bytes()).decode() for name in ('src/spacepdhcg/gtoc12/cli.py','tests/test_gtoc12_gpu_cli.py')}
benchmark=base64.b64encode(Path('build/performance/benchmark_certificate_v710.py').read_bytes()).decode()
script='''from pathlib import Path
import base64,hashlib,json,shutil,subprocess
home=Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus')
base=home/'spacepdhcg-certificate-v709'
root=base/'followup-v710';root.mkdir(exist_ok=False)
for name in ('src','tests'):shutil.copytree(base/'repo'/name,root/'repo'/name,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copyfile(base/'repo/pyproject.toml',root/'repo/pyproject.toml')
'''
script+='files='+repr(files)+'\n'
script+='''for name,encoded in files.items():(root/'repo'/name).write_bytes(base64.b64decode(encoded))
(root/'source-manifest.json').write_text(json.dumps({p.relative_to(root/'repo').as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'repo').rglob('*') if p.is_file()},indent=2))
'''
script+='(root/"benchmark.py").write_bytes(base64.b64decode('+repr(benchmark)+'))\n'
worker='''from pathlib import Path
import fcntl,json,subprocess,time,traceback
root=Path(__file__).resolve().parent;base=root.parent
home=Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus')
namespace={'__file__':str(base/'worker.py')}
exec((base/'worker.py').read_text().split('report=dict(')[0],namespace)
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
'''
script+='(root/"worker.py").write_text('+repr(worker)+')\n'
script+='''with (root/'worker.log').open('x') as log:
    child=subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(root=str(root),pid=child.pid)))
'''
Path('build/performance/launch_certificate_followup_v710.py').write_text(script)
exec(compile(script,'launch-followup-v710','exec'))
