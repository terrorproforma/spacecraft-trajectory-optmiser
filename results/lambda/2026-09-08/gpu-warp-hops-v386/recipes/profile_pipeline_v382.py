from pathlib import Path
import cProfile,pstats,runpy,json,time,hashlib,os
root=Path('build/performance/pipeline-profile-v382');profiler=cProfile.Profile()
wall=time.perf_counter();cpu=time.process_time();profiler.enable()
try:runpy.run_module('spacepdhcg',run_name='__main__')
finally:
 profiler.disable();profiler.dump_stats(str(root/'pipeline.pstats'))
 rows=[]
 for (file,line,name),(primitive,calls,internal,cumulative,callers) in pstats.Stats(profiler).stats.items():
  rows.append(dict(file=file,line=line,name=name,primitive_calls=primitive,calls=calls,internal_seconds=internal,cumulative_seconds=cumulative))
 report=dict(scope='Instrumented complete pipeline profile; not a performance comparison. Search budget increased to avoid profiling overhead truncating the search.',wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,runtime_sha256={key:hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest() for key in ['SPACEPDHCG_GTOC12_CUDA_LIBRARY','SPACEPDHCG_QOCO_LIBRARY']},internal=sorted(rows,key=lambda r:r['internal_seconds'],reverse=True)[:80],cumulative=sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:80])
 (root/'profile-summary.json').write_text(json.dumps(report,indent=2))
