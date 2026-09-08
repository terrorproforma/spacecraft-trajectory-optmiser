from pathlib import Path
import runpy
p=Path('build/performance');s=(p/'run_scaled_pool_campaign_v543.py').read_text()
s=s.replace('scaled-pool-campaign-v543','device-init-campaign-v560').replace('scaled543','device560').replace('build-spacepdhcg-scaled-pool-v539','build-spacepdhcg-device-init-v555')
s=s.replace("env['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1' if enabled else '0'", "env['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='0'\n   env['SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION']='1' if enabled else '0'")
s=s.replace("env['SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL']='1' if enabled else '0'", "env['SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL']='0'")
s=s.replace("['--qoco-ruiz-iterations','2' if enabled else '0']", "['--qoco-ruiz-iterations','0']")
(p/'run_device_initialization_campaign_v560.py').write_text(s)
h=runpy.run_path(str(p/'prepare_scaled_followups.py'));s=h['adapt'](s)
s=s.replace('/home/angus/build-spacepdhcg-device-init-v555/final','/home/ubuntu/spacepdhcg-device-init-v557/core-build/cuda')
s=s.replace("root=Path('build/performance/device-init-campaign-v560');root.mkdir(exist_ok=False)","root=Path('/home/ubuntu/spacepdhcg-device-init-campaign-v561')")
s=s.replace('device560','device561')
code="from pathlib import Path\nimport subprocess,json,shutil\nrepo=Path('/home/ubuntu/spacepdhcg-device-init-v557/repo')\n"
code+="shutil.copy2('/home/ubuntu/spacepdhcg-scaled-pool-v545/repo/build/performance/solver_phase_details.py',repo/'build/performance/solver_phase_details.py')\n"
code+="root=Path('/home/ubuntu/spacepdhcg-device-init-campaign-v561');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(s)+")\n"
code+="with (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid)))\n"
(p/'launch_device_initialization_campaign_v561.py').write_text(code)
