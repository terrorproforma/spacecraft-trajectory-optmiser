from pathlib import Path
p=Path('build/performance')
source=(p/'check_workspace_pool_v476.py').read_text().replace("root=Path('build/performance/workspace-pool-v476')", "root=Path('build/performance/workspace-pool-sanitizers-v481')")
source=source.replace("frozen.mkdir(parents=True,exist_ok=False);",'').replace(";shutil.copy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',core)",'')
source=source.replace("sources=[", "sources=['cpp/cuda/tests/gtoc12_workspace_reuse_test.cu',")
source=source.replace("binary='/home/angus/early-graph-probe-v464'", "binary='/home/angus/workspace-pool-probe-v481'")
source=source.replace("'cpp/cuda/tests/gtoc12_scvx_test.cu','-L'", "'cpp/cuda/tests/gtoc12_workspace_reuse_test.cu','-L'")
a=source.index('jobs=');b=source.index('\nr=dict',a)
source=source[:a]+"jobs=[('compile',compile_cmd),('probe',[binary])]+[(tool,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',binary]) for tool in ['memcheck','synccheck','racecheck']]"+source[b:]
(p/'check_workspace_pool_sanitizers_v481.py').write_text(source)
