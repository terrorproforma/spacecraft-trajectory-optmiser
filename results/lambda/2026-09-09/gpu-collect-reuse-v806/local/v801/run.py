from pathlib import Path
import hashlib,json,os,statistics,subprocess,sys,time,traceback
home=Path.home();root=Path(__file__).resolve().parent
report=dict(pid=os.getpid(),complete=False,success=False,runs=[])
def save():
    p=root/'report.tmp';p.write_text(json.dumps(report,indent=2));p.replace(root/'report.json')
save()
try:
    build=home/'spacepdhcg-collect-v800'
    while not json.loads((build/'report.json').read_text())['complete']:time.sleep(2)
    assert json.loads((build/'report.json').read_text())['success']
    report.update(core_sha256=hashlib.sha256((build/'final/libspacepdhcg_cuda.so').read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((build/'source-manifest.json').read_bytes()).hexdigest());save()
    script=(home/'spacepdhcg-raw-regeneration-v794/run.py').read_text().replace('spacepdhcg-grid-v788','spacepdhcg-collect-v800')
    script=script.replace("order=sorted(fleet.ships,key=lambda s:(cargo[s.ship_id],s.ship_id))", "order=[s for s in fleet.ships if s.ship_id in (10,21)]")
    schedule=[('warm-fresh',False,False),('warm-reuse',True,False)]
    for repeat in range(3):
        for enabled in ([False,True] if repeat%2==0 else [True,False]):schedule.append((f'measured-{repeat}-'+('reuse' if enabled else 'fresh'),enabled,True))
    expected={}
    for name,enabled,measured in schedule:
        out=root/name;out.mkdir();(out/'run.py').write_text(script)
        for f in ('fleet.txt','fit.json'):(out/f).write_bytes((home/'spacepdhcg-regeneration-v792'/f).read_bytes())
        env=dict(os.environ,SPACEPDHCG_TEST_GTOC12_REUSE_COLLECT_DP=str(int(enabled)),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(build/'final/libspacepdhcg_cuda.so'))
        with (out/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(out/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
            report.update(stage=name,child_pid=child.pid);save();status=child.wait()
        r=json.loads((out/'report.json').read_text());assert status==0 and r['complete'] and r['success'],r.get('error')
        runs=[]
        for ship in (10,21):
            p=out/f'ship-{ship:02d}';row=json.loads((p/'report.json').read_text());digest=hashlib.sha256((p/'plans.json').read_bytes()).hexdigest()
            expected.setdefault(ship,digest);assert expected[ship]==digest,(name,ship)
            prior=home/'spacepdhcg-raw-regeneration-v794'/f'ship-{ship:02d}/plans.json'
            assert digest==hashlib.sha256(prior.read_bytes()).hexdigest(),(name,ship,'previous published candidates')
            runs.append(dict(ship=ship,seconds=row['search_seconds'],candidates=row['candidates'],expansions=row['expansions'],plans_sha256=digest,telemetry=row['gpu_telemetry']))
        report['runs'].append(dict(name=name,reuse=enabled,measured=measured,routes=runs));save()
    report['medians']={}
    for ship in (10,21):
        values={str(enabled):statistics.median(row['seconds'] for run in report['runs'] if run['measured'] and run['reuse']==enabled for row in run['routes'] if row['ship']==ship) for enabled in (False,True)}
        report['medians'][ship]=dict(fresh_seconds=values['False'],reuse_seconds=values['True'],speedup=values['False']/values['True'])
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
