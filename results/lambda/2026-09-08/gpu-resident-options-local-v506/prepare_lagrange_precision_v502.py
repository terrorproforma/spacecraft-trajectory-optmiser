from pathlib import Path
p=Path('build/performance')
helper=(p/'diagnose_lagrange_repeat.py').read_text().replace('range(4)','range(12)').replace('step_tolerance=1e-9,objective_tolerance=1e-8','step_tolerance=1e-9,objective_tolerance=1e-8,defect_tolerance=5e-10,clarabel_tolerance=1e-10')
helper=helper.replace('certificate=dataclasses.asdict(certificate),mass=', 'certificate=dataclasses.asdict(certificate),certified=certificate.within_tolerance,mass=')
(p/'diagnose_lagrange_precision_v502.py').write_text(helper)
run=(p/'check_resident_options_v495.py').read_text().replace("root=Path('build/performance/resident-options-v495')", "root=Path('build/performance/lagrange-precision-v502')")
a=run.index('jobs=');b=run.index('\nr=dict',a)
run=run[:a]+"jobs=[('prior',[py,'build/performance/diagnose_lagrange_precision_v502.py',str(root/'prior')]),('candidate',[py,'build/performance/diagnose_lagrange_precision_v502.py',str(root/'candidate')])]"+run[b:]
run=run.replace('   start=time.perf_counter()', "   core_path=Path('/home/angus/build-spacepdhcg-workspace-pool-v490/final/libspacepdhcg_cuda.so') if name=='prior' else core\n   env['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=str(core_path);env['LD_LIBRARY_PATH']=str(core_path.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64'\n   start=time.perf_counter()")
(p/'run_lagrange_precision_v502.py').write_text(run)
remote=(p/'run_lagrange_repeat_v501.py').read_text().replace('lagrange-repeat-v501','lagrange-precision-v503')
a=remote.index(" (root/'diagnose.py').write_text(");b=remote.index("\n for name,library",a)
remote=remote[:a]+" (root/'diagnose.py').write_text("+repr(helper)+")"+remote[b:]
(p/'run_lagrange_precision_v503.py').write_text(remote)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-lagrange-precision-v503');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(remote)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_lagrange_precision_v503.py').write_text(launch)
(p/'status_lagrange_precision_v503.py').write_text((p/'status_lagrange_repeat_v501.py').read_text().replace('lagrange-repeat-v501','lagrange-precision-v503'))
