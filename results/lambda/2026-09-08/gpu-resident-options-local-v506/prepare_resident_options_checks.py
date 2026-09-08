from pathlib import Path
import json
p=Path('build/performance')
sources=['cpp/cuda/include/spacepdhcg/cuda/gtoc12_collection_c_api.h','cpp/cuda/include/spacepdhcg/cuda/orbitweaver_gpu_c_api.h','cpp/cuda/internal/gtoc12_collection_options.h','cpp/cuda/src/gtoc12_collection.cu','cpp/cuda/src/orbitweaver_gpu.cu','src/spacepdhcg/gtoc12/gpu_options.py','src/spacepdhcg/gtoc12/gpu_collection.py','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/lambert.py','src/spacepdhcg/gtoc12/search.py','tests/test_gtoc12_gpu_resident_options.py','tests/test_gtoc12_gpu_collection.py','tests/test_gtoc12_gpu_elements.py']
run=(p/'check_workspace_pool_v490.py').read_text().replace('workspace-pool-v490','resident-options-v494')
a=run.index('sources=');b=run.index('\nfor name in sources:',a);run=run[:a]+'sources='+repr(sources)+run[b:]
a=run.index('jobs=');b=run.index('\nr=dict',a)
run=run[:a]+"jobs=[('pytest',[py,'-c',boot,'tests/test_gtoc12_gpu_resident_options.py','tests/test_gtoc12_gpu_collection.py','tests/test_gtoc12_gpu_elements.py','-x','-s','-q'])]"+run[b:]
a=run.index(" result=json.loads((root/'output/run_report.json')");b=run.index(" r['complete']=True",a);run=run[:a]+run[b:]
(p/'check_resident_options_v494.py').write_text(run)
(p/'resident-options-sources.json').write_text(json.dumps(sources,indent=2))
