from pathlib import Path
p=Path('build/performance');helper=(p/'diagnose_lagrange_repeat.py').read_text()
run=(p/'run_resident_options_v500.py').read_text().replace('resident-options-v500','lagrange-repeat-v501')
a=run.index(" shutil.copytree(");b=run.index(" cmake=",a)
run=run[:a]+" repo=Path('/home/ubuntu/spacepdhcg-resident-options-v500/repo')\n report['source_base']='resident-options-v500/repo'\n"+run[b:]
a=run.index(' boot=');b=run.index(" report['complete']=True",a)
run=run[:a]+" (root/'diagnose.py').write_text("+repr(helper)+")\n for name,library in [('prior',Path('/home/ubuntu/spacepdhcg-workspace-pool-v491/core-build/cuda/libspacepdhcg_cuda.so')),('candidate',core)]:\n  case=env.copy();case['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=str(library);case['LD_LIBRARY_PATH']=str(library.parent)+':'+env['LD_LIBRARY_PATH']\n  run(name,[py,str(root/'diagnose.py'),str(root/name)],900,environment=case)\n"+run[b:]
(p/'run_lagrange_repeat_v501.py').write_text(run)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-lagrange-repeat-v501');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(run)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_lagrange_repeat_v501.py').write_text(launch)
(p/'status_lagrange_repeat_v501.py').write_text((p/'status_resident_options_v500.py').read_text().replace('resident-options-v500','lagrange-repeat-v501'))
