from pathlib import Path
p=Path('build/performance')
source=(p/'run_workspace_pool_v484.py').read_text()
source=source.replace('workspace-pool-v484','workspace-pool-v487').replace('workspace_pool484_','workspace_pool487_')
source=source.replace("shutil.copytree('/home/ubuntu/spacepdhcg-early-graph-v471/repo'", "shutil.copytree('/home/ubuntu/spacepdhcg-workspace-pool-v484/repo'")
a=source.index(" with tarfile.open(");b=source.index(' source=json.loads',a);source=source[:a]+source[b:]
source=source.replace("core=root/'core-build/cuda/libspacepdhcg_cuda.so'", "core=Path('/home/ubuntu/spacepdhcg-workspace-pool-v484/core-build/cuda/libspacepdhcg_cuda.so')")
a=source.index(" run('configure'");b=source.index(" env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']",a)
source=source[:a]+" report['runtime_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]};save()\n"+source[b:]
source=source.replace("  env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='0' if candidate else '0'", "  env['SPACEPDHCG_TEST_GTOC12_QOCO_POOL']='1' if candidate else '0'")
source=source.replace("  r=json.loads((root/name/'output/run_report.json').read_text());", "  calls=json.loads((root/name/'calls.json').read_text());creations=sum(max(p['workspace_creations'] for p in c['solver_reports']) for c in calls);assert (creations<47) if candidate else (creations==47)\n  r=json.loads((root/name/'output/run_report.json').read_text());")
(p/'run_workspace_pool_v487.py').write_text(source)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-workspace-pool-v487');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(source)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_workspace_pool_v487.py').write_text(launch)
(p/'status_workspace_pool_v487.py').write_text((p/'status_workspace_pool_v484.py').read_text().replace('workspace-pool-v484','workspace-pool-v487'))
