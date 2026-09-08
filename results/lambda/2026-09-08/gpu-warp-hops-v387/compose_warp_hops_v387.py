from pathlib import Path
s=Path('build/performance/run_fleet_recovery_v381.py').read_text()
s=s.replace("root=Path('/home/ubuntu/spacepdhcg-fleet-recovery-v381')","root=Path('/home/ubuntu/spacepdhcg-warp-hops-v387')")
s=s.replace("shutil.copytree('/home/ubuntu/spacepdhcg-harvest-window-v378/repo'","shutil.copytree('/home/ubuntu/spacepdhcg-fleet-recovery-v381/repo'")
s=s.replace('/tmp/fleet-recovery-v381.tar.gz','/tmp/warp-hops-v387.tar.gz').replace('recovery-source-sha256.json','warp-source-sha256.json')
s=s.replace('save()\ntry:',"save()\nlock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');report['stage']='waiting_for_gpu';save()\nfcntl.flock(lock,fcntl.LOCK_EX)\ntry:",1)
s=s.replace(" with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:\n  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)"," if True:")
s=s.replace("tests=['tests/test_gtoc12_refinement_queue.py'","tests=['tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_gpu_lambert.py','tests/test_gtoc12_refinement_queue.py'")
start=s.index("  cmd=[py,'-c',cli,'gtoc12','run'");end=s.index(" report['complete']=True",start)
s=s[:start]+'''  run('synccheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','synccheck','--error-exitcode','99',py,'-c',boot,tests[0],'-q'],300)
  baseline_core=Path('/home/ubuntu/spacepdhcg-fleet-recovery-v381/core-build/cuda/libspacepdhcg_cuda.so')
  micro_boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/hop_micro_v384.py',run_name='__main__')"
  for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:
   lib=core if candidate else baseline_core
   selected=dict(env,SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(lib));selected['LD_LIBRARY_PATH']=str(lib.parent)+':'+str(qoco.parent)+':'+runtime+'/lib:/usr/local/cuda/lib64'
   run('micro_'+name,[py,'-c',micro_boot,str(root/('micro_'+name))],environment=selected)
  for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:
   lib=core if candidate else baseline_core
   selected=dict(env,SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(lib));selected['LD_LIBRARY_PATH']=str(lib.parent)+':'+str(qoco.parent)+':'+runtime+'/lib:/usr/local/cuda/lib64'
   cmd=[py,'-c',cli,'gtoc12','run','--run-id','warp387_'+name,'--output',str(root/name/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','600','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
   run(name,cmd,900,environment=selected)
   r=json.loads((root/name/'output/run_report.json').read_text());assert r['best']['official']['ok'] and r['best']['independent']['ok']
   report['campaigns'].append(dict(name=name,candidate=candidate,seconds=r['wall_seconds_total'],score=r['best']['independent']['weighted_score_fixed_bonus_kg'],screening=r['screening']));save()
'''+s[end:]
compile(s,'run387','exec');Path('build/performance/run_warp_hops_v387.py').write_text(s)
s=Path('build/performance/prepare_fleet_recovery_v381.py').read_text();start=s.index('files=');end=s.index('\nmanifest=',start)
s=s[:start]+"files=['cpp/cuda/src/orbitweaver_gpu.cu','tests/test_gtoc12_gpu_hops.py','build/performance/hop_micro_v384.py']"+s[end:]
s=s.replace('recovery-v380/recovery-source-sha256.json','warp-hops-v385/warp-source-sha256.json').replace('fleet-recovery-v381','warp-hops-v387').replace('recovery-source-sha256.json','warp-source-sha256.json').replace('run_fleet_recovery_v381','run_warp_hops_v387').replace('launch_fleet_recovery_v381','launch_warp_hops_v387')
compile(s,'prepare387','exec');Path('build/performance/prepare_warp_hops_v387.py').write_text(s)
