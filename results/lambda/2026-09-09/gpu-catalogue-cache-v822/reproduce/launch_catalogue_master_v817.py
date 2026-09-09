from pathlib import Path
import hashlib,json,shutil,subprocess
home=Path.home();root=home/'spacepdhcg-catalogue-master-v817';root.mkdir()
origin=Path('results/local/2026-09-09/mass-budgeted-frontier-v629/Result.txt');expected='765cb7ef97d38926317dbbf4ae8700626dffec06c48af3d6f68dae47c154f3da'
assert hashlib.sha256(origin.read_bytes()).hexdigest()==expected
report=json.loads(Path('results/local/2026-09-09/mass-budgeted-frontier-v629/report.json').read_text());assert report['qualified'] and report['result_sha256']==expected
shutil.copyfile(origin,root/'incumbent.txt');(root/'incumbent-report.json').write_text(json.dumps(report,indent=2))
script=(home/'spacepdhcg-collect-master-v804/run.py').read_text().replace('spacepdhcg-collect-v800','spacepdhcg-catalogue-v808').replace('spacepdhcg-collect-refine-v803','spacepdhcg-integrated-refine-v815')
script=script.replace("original=home/'spacepdhcg-collect-fleet-v802/fleet.txt'", "original=root/'incumbent.txt'")
script=script.replace("    started=time.perf_counter();report['stage']='pack_certified'", "    benchmark=home/'spacepdhcg-catalogue-bench-v816/report.json'\n    while not benchmark.exists() or not json.loads(benchmark.read_text())['complete']:time.sleep(2)\n    assert json.loads(benchmark.read_text())['success']\n    assert hashlib.sha256((root/'incumbent.txt').read_bytes()).hexdigest()=="+repr(expected)+"\n    report['incumbent_sha256']="+repr(expected)+"\n    report['source_manifest_sha256']=hashlib.sha256((home/'spacepdhcg-catalogue-v808/source-manifest.json').read_bytes()).hexdigest()\n    started=time.perf_counter();report['stage']='pack_certified'")
script=script.replace("if report['improved']:","if report['qualified']:").replace("run_id='gpu_collect_reuse_v804',commit='ccf5de37'", "run_id='gpu_catalogue_integrated_v817',commit='5f698731'")
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-collect-master-v804/launch.py').read_text().replace('spacepdhcg-collect-master-v804','spacepdhcg-catalogue-master-v817').replace('spacepdhcg-collect-v800','spacepdhcg-catalogue-v808');(root/'launch.py').write_text(launch)
target='ubuntu@192.222.55.229';key='/tmp/traj-key.pem'
subprocess.run(['ssh','-i',key,'-o','BatchMode=yes',target,'mkdir /home/ubuntu/spacepdhcg-catalogue-master-v817'],check=True,timeout=20)
subprocess.run(['scp','-q','-i',key,*[str(root/p) for p in ('run.py','launch.py','incumbent.txt','incumbent-report.json')],target+':/home/ubuntu/spacepdhcg-catalogue-master-v817/'],check=True,timeout=55)
subprocess.run(['ssh','-i',key,'-o','BatchMode=yes',target,'python3 /home/ubuntu/spacepdhcg-catalogue-master-v817/launch.py'],check=True,timeout=20)
exec(launch)
