from pathlib import Path
import json,subprocess
root=Path('/home/ubuntu/spacepdhcg-joint-selection-v632')
for name in ('report.json','validation-v631/pytest.log','validation-v631/report.json','benchmark-v634/report.json','benchmark-v634/scalar-vs-batched.log','benchmark-v634/selection.log'):
    path=root/name
    if path.exists():print(name,path.read_text())
print(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True))
