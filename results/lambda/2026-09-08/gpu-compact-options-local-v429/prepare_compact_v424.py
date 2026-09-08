from pathlib import Path
base = Path('build/performance/check_compact_options_v423.py').read_text()
base = base.replace("root=Path('build/performance/compact-options-v423')", "root=Path('build/performance/compact-options-v424')")
base = base.replace(";frozen.mkdir(parents=True,exist_ok=False)", "")
base = base.replace("shutil.copy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',core)", "")
base = base.replace("'src/spacepdhcg/gtoc12/search.py']", "'src/spacepdhcg/gtoc12/search.py','tests/test_gtoc12_gpu_elements.py']")
start = base.index("jobs=[")
end = base.index("\nr=dict", start)
base = base[:start] + "jobs=[('pytest',[py,'-c',boot,'tests/test_gtoc12_gpu_elements.py','tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_search2.py','-q'])]" + base[end:]
Path('build/performance/check_compact_v424.py').write_text(base)

base=Path('build/performance/run_options_once_v421.py').read_text()
base=base.replace('options-once-v421','compact-options-v425').replace('build-spacepdhcg-stationary-v411','build-spacepdhcg-compact-options-v423')
start=base.index('names=');end=base.index('\nreport=',start)
base=base[:start]+"names=['src/spacepdhcg/gtoc12/search.py','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/lambert.py','tests/test_gtoc12_gpu_elements.py','cpp/cuda/src/orbitweaver_gpu.cu','cpp/cuda/include/spacepdhcg/cuda/orbitweaver_gpu_c_api.h']"+base[end:]
base=base.replace("runpy.run_path('build/performance/options_compare_boot_v421.py',run_name='__main__')", "runpy.run_module('spacepdhcg',run_name='__main__')")
base=base.replace("env['SPACEPDHCG_BENCHMARK_SEARCH_BASELINE']='0' if candidate else '1'", "env['SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS']='1' if candidate else '0'")
base=base.replace("   env['SPACEPDHCG_BENCHMARK_SEARCH_SOURCE']=str(Path('build/performance/options-baseline-v421.py').resolve())\n", "")
base=base.replace('options421_', 'compact425_')
Path('build/performance/run_compact_v425.py').write_text(base)
