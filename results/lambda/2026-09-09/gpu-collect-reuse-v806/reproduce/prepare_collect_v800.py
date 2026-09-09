from pathlib import Path
p=Path('build/performance')
worker=(p/'worker_grid_v786.py').read_text()
a=worker.index("        run('pytest'");b=worker.index("    report['success']",a)
worker=worker[:a]+'''        tests=['tests/test_gtoc12_gpu_collect_dp.py','tests/test_gtoc12_gpu_collect_workspace.py','tests/test_gtoc12_gpu_collect_tables.py','tests/test_gtoc12_gpu_resident_collect_tables.py']
        run('pytest',[py,'-c',boot,'-q',*tests])
        for mode in ('memcheck','racecheck','synccheck'):
            run(mode,[cuda+'/bin/compute-sanitizer','--tool',mode,'--error-exitcode','86',py,'-c',boot,'-q','tests/test_gtoc12_gpu_collect_workspace.py','tests/test_gtoc12_gpu_resident_collect_tables.py'])
''' +worker[b:]
(p/'worker_collect_v800.py').write_text(worker)
build=(p/'build_grid_v786.py').read_text().replace('spacepdhcg-grid-v786','spacepdhcg-collect-v800').replace('worker_grid_v786','worker_collect_v800')
a=build.index('owned=');b=build.index('\nfor name in owned:',a)
owned=['cpp/cuda/src/gtoc12_collect_dp.cu','cpp/cuda/include/spacepdhcg/cuda/gtoc12_collect_dp_c_api.h','src/spacepdhcg/gtoc12/gpu_collect_dp.py','tests/test_gtoc12_gpu_collect_workspace.py','tests/test_gtoc12_gpu_resident_collect_tables.py']
build=build[:a]+'owned='+repr(owned)+build[b:]
needle="(root/'source-manifest.json').write_text"
fixture="results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json"
build=build.replace(needle,"fixture="+repr(fixture)+"\n(repo/fixture).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(fixture,repo/fixture)\n"+needle)
(p/'build_collect_v800.py').write_text(build)
(p/'send_collect_v800.py').write_text((p/'send_grid_v786.py').read_text().replace('spacepdhcg-grid-v786','spacepdhcg-collect-v800').replace('grid-source-v786','collect-source-v800').replace('launch_grid_h100_v786','launch_collect_h100_v800'))
(p/'status_collect_v800.py').write_text((p/'status_grid_v786.py').read_text().replace('spacepdhcg-grid-v786','spacepdhcg-collect-v800'))
