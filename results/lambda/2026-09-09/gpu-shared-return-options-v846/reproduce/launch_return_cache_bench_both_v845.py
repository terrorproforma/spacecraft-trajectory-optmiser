from pathlib import Path
import subprocess
home=Path.home();root=home/'spacepdhcg-return-cache-bench-v845';root.mkdir()
script=(home/'spacepdhcg-admission-bench-v840/run.py').read_text()
script=script.replace('spacepdhcg-admission-v839','spacepdhcg-return-cache-v844').replace('SPACEPDHCG_TEST_GTOC12_GPU_ADMISSION','SPACEPDHCG_TEST_GTOC12_RETAIN_RETURN_OPTIONS')
script=script.replace('CUDA beam admission; CUDA expansion, catalogue and DP reuse enabled in both modes','Shared resident return options; CUDA admission, expansion, catalogue and DP reuse enabled in both modes')
script=script.replace('spacepdhcg-expansion-bench-v831/warm-reuse','spacepdhcg-admission-bench-v840/warm-reuse')
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-admission-bench-v840/launch.py').read_text().replace('spacepdhcg-admission-bench-v840','spacepdhcg-return-cache-bench-v845').replace('spacepdhcg-admission-v839','spacepdhcg-return-cache-v844')
(root/'launch.py').write_text(launch);exec(launch)
