from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import json
import os
import statistics
import sys
import time
import numpy as np

sys.path.insert(0,str(Path.cwd()/'tests'))
import test_gtoc12_gpu_joint as oracle
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.gpu_joint import evaluate_joint
from spacepdhcg.gtoc12.lambert import using_lambert_backend

os.environ.update(SPACEPDHCG_TEST_GTOC12_JOINT_BATCH='1',SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY='1',SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH='1')
data=(load_catalogue(),load_bonus_table())
report={'complete':False,'samples':[],'comparisons':[]}
destination=Path(sys.argv[1])
def save():destination.write_text(json.dumps(report,indent=2))
save()
with using_lambert_backend('cuda',maximum_batch_size=97) as gpu:
    for ship in ('ship-01.json','ship-02.json'):
        template,visits,arr,dep=oracle.incumbent.__wrapped__(SimpleNamespace(param=ship),data)
        rows=oracle._moves(visits,arr,dep,3.0)
        values=evaluate_joint(deepcopy(template),visits,*rows)
        worst=min((i for i,v in enumerate(values) if v.feasible),key=lambda i:values[i].objective)
        for case,a,d in [('incumbent',arr,dep),('perturbed',rows[0][worst],rows[1][worst])]:
            reference=None
            for repeat in range(7):
                for mode in ((0,1) if repeat%2==0 else (1,0)):
                    os.environ['SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SEARCH']=str(mode)
                    joint=deepcopy(template);before=dict(gpu.telemetry)
                    start=time.perf_counter();out=joint.optimise_epochs(visits,a,d,mesh=(8.0,3.0,1.0),max_moves=10);seconds=time.perf_counter()-start
                    if reference is None:reference=out
                    np.testing.assert_array_equal(out[0],reference[0]);np.testing.assert_array_equal(out[1],reference[1])
                    oracle._same(out[2],reference[2],atol=0,rtol=0);assert out[3]==reference[3]
                    report['samples'].append(dict(ship=ship,case=case,repeat=repeat,warmup=repeat==0,mode=mode,seconds=seconds,moves=out[3],objective=out[2].objective,weighted_kg=out[2].weighted_kg,evaluations=joint.evaluations-template.evaluations,arrivals=out[0].tolist(),departures=out[1].tolist(),telemetry={k:v-before.get(k,0) for k,v in gpu.telemetry.items() if isinstance(v,(int,float))}))
                    save()
            group=[r for r in report['samples'] if r['ship']==ship and r['case']==case and not r['warmup']]
            medians={str(mode):statistics.median(r['seconds'] for r in group if r['mode']==mode) for mode in (0,1)}
            report['comparisons'].append(dict(ship=ship,case=case,medians=medians,ratio=medians['0']/medians['1'],all_final_values_exact=True))
report['complete']=True;save();print(json.dumps(report['comparisons']),flush=True)
