from pathlib import Path
import subprocess
baseline=Path('build/performance/options-baseline-v421.py')
baseline.write_bytes(subprocess.check_output(['git','show','18e10cb042e6ca208d0c47be24da944aebbd2c50:src/spacepdhcg/gtoc12/search.py']))
source=Path('build/performance/run_paired_ephemerides_v416.py').read_text()
source=source.replace('paired-ephemerides-v416','options-once-v421').replace('paired416_','options421_')
source=source.replace("names=['src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/lambert.py','src/spacepdhcg/gtoc12/search.py','tests/test_gtoc12_gpu_elements.py']","names=['src/spacepdhcg/gtoc12/search.py','build/performance/options-baseline-v421.py','build/performance/options_compare_boot_v421.py']")
source=source.replace("env['SPACEPDHCG_TEST_GTOC12_PAIRED_EPHEMERIDES']='1' if candidate else '0'","env['SPACEPDHCG_BENCHMARK_SEARCH_BASELINE']='0' if candidate else '1'\n   env['SPACEPDHCG_BENCHMARK_SEARCH_SOURCE']=str(Path('build/performance/options-baseline-v421.py').resolve())")
source=source.replace("runpy.run_module('spacepdhcg',run_name='__main__')","runpy.run_path('build/performance/options_compare_boot_v421.py',run_name='__main__')")
source=source.replace("assert r['best']['official']['ok'] and r['best']['independent']['ok']","assert r['best']['accepted'] and r['best']['official']['ok'] and r['best']['independent']['ok']")
Path('build/performance/run_options_once_v421.py').write_text(source)
