from pathlib import Path
root=Path('build/performance/step-only-v353');root.mkdir(exist_ok=False)
for name in ['arithmetic.cuh','old_step.cuh','new_step.cuh','inputs.bin']:(root/name).write_bytes((Path('build/performance/cone-step-v345')/name).read_bytes())
(root/'probe.cu').write_text(Path('build/performance/cone-step-v345/probe.cu').read_text().replace('cone determinants evaluated','SOC step bounds evaluated'))
s=Path('build/performance/run_nt_lambda_v350.py').read_text().replace('spacepdhcg-nt-v350','spacepdhcg-step-v353').replace('nt-normalization-v347','step-only-v353')
a=s.index("a=text.index('__global__ void compute_nt_scaling_kernel')");b=s.index("a=text.index('__device__ QOCOFloat soc_step_length_dev(')",a)
s=s[:a]+'text=\'#include "qoco_cone_arithmetic.cuh"\\n\'+text\n'+s[b:]
s=s.replace("(overlay/'new_step.cuh')","(patch/'new_step.cuh')").replace('nt-snapshot-v348','step-snapshot-v351').replace("'nt_step'","'step_only'").replace("'nt_step_repeat'","'step_only_repeat'")
anchor=" report['complete']=True"
extra=''' # Whole missions are separate jobs which acquire the GPU lock themselves.
 template=Path('/home/ubuntu/spacepdhcg-nonfinite-confirm-v341/v342/run.py').read_text()
 report['campaigns']=[];save()
 for version,candidate in [(354,True),(355,False),(356,False),(357,True)]:
  path=root/f'v{version}';path.mkdir(exist_ok=False)
  recipe=template.replace('/home/ubuntu/spacepdhcg-nonfinite-confirm-v341/v342',str(path)).replace('gpu_nonfinite_342','gpu_step_'+str(version))
  if candidate:recipe=recipe.replace('/home/ubuntu/spacepdhcg-nonfinite-ir-v337/final/libqoco.so',str(lib))
  # Existing template retains its frozen core/source provenance; add the exact candidate cone source.
  recipe=recipe.replace("report['kernel_source_sha256']=", "report['step_candidate']="+repr(candidate)+"\\nreport['step_cone_source_sha256']="+repr(report['cone_source_sha256'] if candidate else None)+"\\nreport['kernel_source_sha256']=",1)
  (path/'run.py').write_text(recipe)
  with (path/'runner.log').open('x') as log:
   child=subprocess.Popen(['python3',str(path/'run.py')],stdout=log,stderr=subprocess.STDOUT)
   row=dict(version=version,candidate=candidate,pid=child.pid,complete=False);report['campaigns'].append(row);save()
   row['returncode']=child.wait(timeout=1900)
  run_report=json.loads((path/'report.json').read_text());assert run_report['complete'] and run_report['returncode']==0
  output=json.loads((path/'output/run_report.json').read_text())
  row.update(complete=True,seconds=output['wall_seconds_total'],score=output['best']['score_kg'],official=output['best']['official']['ok'],independent=output['best']['independent']['ok']);save();print(row,flush=True)
'''
s=s.replace(anchor,extra+anchor)
s=s.replace("\na=text.index", "\n a=text.index")
compile(s,'run_step_lambda_v353.py','exec')
Path('build/performance/run_step_lambda_v353.py').write_text(s)
d=Path('build/performance/dispatch_nt_lambda_v350.py').read_text().replace('nt-v350','step-v353').replace("('nt-normalization-v347',['probe.cu','arithmetic.cuh','old_kernel.cuh','new_kernel.cuh','inputs.bin'])","('step-only-v353',['probe.cu','arithmetic.cuh','old_step.cuh','new_step.cuh','inputs.bin'])").replace('nt-snapshot-v348','step-snapshot-v351').replace('run_nt_lambda_v350.py','run_step_lambda_v353.py')
d=d.replace(" t.add('build/performance/cone-step-v345/new_step.cuh',arcname='new_step.cuh')\n",'')
Path('build/performance/dispatch_step_lambda_v353.py').write_text(d)
p=Path('build/performance/poll_nt_lambda_v350.py').read_text().replace('spacepdhcg-nt-v350','spacepdhcg-step-v353')
p+="\nprint('campaigns',r.get('campaigns'))\n"
Path('build/performance/poll_step_lambda_v353.py').write_text(p)
