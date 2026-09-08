from pathlib import Path
import subprocess,json,time
root=Path('build/performance/soc-step-header-v358');root.mkdir(exist_ok=False)
source='cpp/cuda/tests/qoco_soc_step_probe.cu';nvcc='/usr/local/cuda-12.8/bin/nvcc'
reports=[]
def run(name,cmd,expected=0):
 start=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
 (root/(name+'.log')).write_text(r.stdout+r.stderr);reports.append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-start));(root/'report.json').write_text(json.dumps(reports,indent=2));print(name,r.returncode,(r.stdout+r.stderr)[-600:],flush=True);assert r.returncode==expected
old='/home/angus/qoco-soc-step-old-v358';new='/home/angus/qoco-soc-step-probe-v352'
run('old-build',[nvcc,'-O3','-std=c++17','-arch=sm_120','--default-stream','per-thread',source,'-Ibuild/performance/cone-step-v345','-DSPACEPDHCG_TEST_SOC_STEP_HEADER="old_step.cuh"','-o',old])
run('old-regression',[old],2);run('new',[new])
for tool in ['memcheck','initcheck','racecheck','synccheck']:
 run(tool,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',new])
