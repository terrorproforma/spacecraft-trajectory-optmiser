from pathlib import Path
runner=Path('results/lambda/2026-09-08/gpu-collect-tables-v290/v291/run.py').read_text()
runner=runner.replace('collect-tables-campaign-v291','collect-resident-campaign-v299').replace('gpu_collect_tables_campaign_v291','gpu_collect_resident_campaign_v299').replace("integrated=Path('/home/ubuntu/spacepdhcg-collect-tables-v290')","integrated=Path('/home/ubuntu/spacepdhcg-collect-resident-v298')")
runner=runner.replace("core=Path('/home/ubuntu/spacepdhcg-collect-dp-v279/core-build/cuda/libspacepdhcg_cuda.so')","core=integrated/'core-build/cuda/libspacepdhcg_cuda.so'")
runner=runner.replace("source_base_commit='a817f164'","source_base_commit='f07e82c5'").replace("['CUDA orbital-element collection tables']","['resident float32 collection tables and device gather into DP']")
root='/home/ubuntu/spacepdhcg-collect-resident-campaign-v299'
launch="from pathlib import Path\nimport subprocess\nroot=Path("+repr(root)+");root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(runner)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(p.pid)\n"
Path('build/performance/launch_resident_campaign_v299.py').write_text(launch)
Path('build/performance/poll_resident_campaign_v299.py').write_text(Path('build/performance/poll_collect_campaign_v291.py').read_text().replace('collect-tables-campaign-v291','collect-resident-campaign-v299'))
