from pathlib import Path
import subprocess
home=Path.home();root=home/'spacepdhcg-return-cache-leaks-v847';root.mkdir()
script=(home/'spacepdhcg-admission-leaks-v842/run.py').read_text().replace('spacepdhcg-admission-v839','spacepdhcg-return-cache-v844').replace('spacepdhcg-admission-bench-v840','spacepdhcg-return-cache-bench-v845')
script=script.replace("'tests/test_gtoc12_gpu_expansion.py','-k','not layout and not empty'", "'tests/test_gtoc12_gpu_expansion.py','tests/test_gtoc12_gpu_return_cache.py','tests/test_gtoc12_gpu_resident_options.py','-k','not layout and not empty_screening'")
(root/'run.py').write_text(script)
with (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
