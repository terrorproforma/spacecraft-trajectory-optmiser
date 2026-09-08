from pathlib import Path
p=Path('build/performance/prepare_warp_hops_v387.py');s=p.read_text();s=s.replace("'tests/test_gtoc12_gpu_hops.py','build/performance/hop_micro_v384.py'","'tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_run_refinement.py','build/performance/hop_micro_v384.py'");p.write_text(s)
