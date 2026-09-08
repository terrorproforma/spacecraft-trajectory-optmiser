"""Matched CPU/GPU epoch generation including resident geometry and selected output."""
from pathlib import Path
import importlib.util,json,math,os,statistics,sys,time
import numpy as np
from spacepdhcg.gtoc12.gpu_joint import evaluate_mesh
source=Path.cwd()
spec=importlib.util.spec_from_file_location('joint_benchmark',source/'build/performance/joint-benchmark-v593/benchmark_joint.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
output=Path(sys.argv[1]);assert not output.exists()
report=dict(complete=False,scope=__doc__,core=b.file_record(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']),cases=[],controller=[])
catalogue,bonus=b.load_catalogue(),b.load_bonus_table()
os.environ[b.JOINT_FLAG]='1';os.environ['SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY']='1';os.environ['SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION']='1'
archive=source/'results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources'
with b.using_lambert_backend('cuda') as gpu:
 for ship in ('ship-01.json','ship-02.json'):
  template,visits,arr,dep,fixture=b.prepare_fixture(archive/ship,catalogue,bonus,True)
  arrivals,departures=b.matrices(visits,arr,dep,3.)
  warm=b.warm_cache(gpu,template,visits,np.vstack((arr,arrivals)),np.vstack((dep,departures)))
  for condition,cache in (('cold',{}),('warm',warm)):
   expected=None;case=dict(ship=ship,visits=len(visits),candidates=len(arrivals),cache=condition,runs=[],fixture=fixture)
   for block in range(6):
    for mode in (0,1,1,0):
     joint=b.clone(template,cache);before=b.counters(gpu)
     start=time.perf_counter_ns()
     if mode:
      result=evaluate_mesh(joint,visits,arr,dep,3.,-math.inf)
     else:
      a,d=b.matrices(visits,arr,dep,3.)
      index,value=b.evaluate_joint(joint,visits,a,d,minimum_objective=-math.inf)
      result=(a[index],d[index],value)
     seconds=(time.perf_counter_ns()-start)*1e-9
     if expected is None:expected=result
     np.testing.assert_array_equal(result[0],expected[0]);np.testing.assert_array_equal(result[1],expected[1])
     check=b.Checks(0.,0.);check.evaluation(result[2],expected[2],'winner');assert not check.errors,check.errors
     if block:case['runs'].append(dict(mode=mode,seconds=seconds,telemetry=b.counter_delta(before,gpu)))
   medians={str(mode):statistics.median(r['seconds'] for r in case['runs'] if r['mode']==mode) for mode in (0,1)}
   case.update(median_seconds=medians,speedup=medians['0']/medians['1'],exact_winner_parity=True)
   report['cases'].append(case);output.write_text(json.dumps(b.plain(report),indent=2,allow_nan=False));print(ship,condition,case['speedup'],flush=True)
  values=b.evaluate_joint(b.clone(template,{}),visits,arrivals,departures)
  worst=min((i for i,v in enumerate(values) if v.feasible),key=lambda i:values[i].objective)
  for mode in (0,1):
   os.environ['SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH']=str(mode)
   joint=b.clone(template,{})
   start=time.perf_counter();result=joint.optimise_epochs(visits,arrivals[worst],departures[worst],mesh=(3.,1.),max_moves=3)
   record=dict(ship=ship,mode=mode,seconds=time.perf_counter()-start,arrivals=result[0],departures=result[1],evaluation=result[2],accepted_moves=result[3],evaluations=joint.evaluations,lambert_requests=joint.lambert_evaluations)
   assert result[3]>0
   if mode==0:reference=result
   else:
    np.testing.assert_array_equal(result[0],reference[0]);np.testing.assert_array_equal(result[1],reference[1]);assert result[3]==reference[3]
    check=b.Checks(0.,0.);check.evaluation(result[2],reference[2],'controller');assert not check.errors,check.errors
   report['controller'].append(record)
report['complete']=True;output.write_text(json.dumps(b.plain(report),indent=2,allow_nan=False))
