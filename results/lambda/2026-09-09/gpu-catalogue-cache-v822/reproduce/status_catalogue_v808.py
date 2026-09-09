from pathlib import Path
import json,os,subprocess
for dirname in ('spacepdhcg-catalogue-v808','spacepdhcg-catalogue-bench-v809','spacepdhcg-integrated-refine-v810'):
    root=Path.home()/dirname
    if not (root/'report.json').exists():continue
    r=json.loads((root/'report.json').read_text());info={k:r[k] for k in ('complete','success','stage','pid','child_pid','error','native_solves','score_kg','qualified','seconds') if k in r}
    for key in ('pid','child_pid'):
        if key in r:info[key+'_live']=Path('/proc',str(r[key])).exists()
    if 'stages' in r:info['finished_stages']=[(s['name'],s['code']) for s in r['stages']]
    if 'runs' in r:info['runs']=len(r['runs'])
    if 'attempts' in r:info['routes']=len(r['attempts'])
    print(dirname,json.dumps(info))
    if r.get('complete') and not r.get('success'):
        for name in (r.get('stage','')+'.log','worker.log'):
            if (root/name).exists():print((root/name).read_text()[-4000:])
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
