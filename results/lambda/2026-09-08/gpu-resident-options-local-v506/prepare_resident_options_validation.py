from pathlib import Path
import json
p=Path('build/performance')
run=(p/'check_resident_options_v494.py').read_text().replace("root=Path('build/performance/resident-options-v494')","root=Path('build/performance/resident-options-v495')")
a=run.index('frozen=');b=run.index('\nsources=',a)
run=run[:a]+"frozen=Path('/home/angus/build-spacepdhcg-resident-options-v494/final');core=frozen/'libspacepdhcg_cuda.so'"+run[b:]
run=run.replace("sources=[", "sources=['cpp/cuda/tests/gtoc12_resident_options_test.cu',")
run=run.replace("binary='/home/angus/workspace-pool-probe-v490'", "binary='/home/angus/resident-options-probe-v495'")
run=run.replace('cpp/cuda/tests/gtoc12_workspace_reuse_test.cu','cpp/cuda/tests/gtoc12_resident_options_test.cu')
a=run.index('jobs=');b=run.index('\nr=dict',a)
run=run[:a]+"jobs=[('compile',compile_cmd),('probe',[binary])]+[(tool,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',binary]) for tool in ['memcheck','synccheck','racecheck']]+[('replay',[binary,'build/performance/resident-options-fixture.bin'])]"+run[b:]
(p/'check_resident_options_v495.py').write_text(run)
# A same-binary full campaign pair, using the established command and runner.
run=(p/'run_profile_v492.py').read_text().replace('pipeline-profile-v492','resident-options-campaign-v496').replace('build-spacepdhcg-workspace-pool-v490','build-spacepdhcg-resident-options-v494')
run=run.replace("runpy.run_path('build/performance/profile_v492.py',run_name='__main__')", "runpy.run_module('spacepdhcg',run_name='__main__')")
run=run.replace("'build/performance/profile_v492.py'", "'src/spacepdhcg/gtoc12/gpu_options.py'")
a=run.index('  start=time.perf_counter()');b=run.index("  r['complete']=True",a)
run=run[:a]+'''  r['campaigns']=[]
  for name,enabled in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:
   env['SPACEPDHCG_TEST_GTOC12_RESIDENT_OPTIONS']='1' if enabled else '0'
   command=cmd.copy();command[command.index('--output')+1]=str(root/name/'output')
   start=time.perf_counter()
   with (root/(name+'.log')).open('x') as log:
    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);r.update(child_pid=child.pid,stage=name);save();code=child.wait(timeout=900)
   seconds=time.perf_counter()-start
   assert code==0,(name,code)
   result=json.loads((root/name/'output/run_report.json').read_text());assert result['best']['accepted'] and result['best']['official']['ok'] and result['best']['independent']['ok']
   screening=result['screening'];assert bool(screening.get('resident_option_builds',0))==enabled
   if enabled:assert screening.get('resident_option_read_bytes',0)==0 and screening['collection_option_upload_bytes']==0
   r['campaigns'].append(dict(name=name,candidate=enabled,process_seconds=seconds,cli_seconds=result['wall_seconds_total'],score=result['best']['independent']['weighted_score_fixed_bonus_kg'],screening=screening));save()
'''+run[b:]
(p/'run_resident_options_v496.py').write_text(run)
