from pathlib import Path
root=Path('/home/ubuntu/spacepdhcg-joint-selection-v632')
for name in ('compatibility-runner.log','compatibility-validation-v640/report.json','archive-runner.log'):
    p=root/name
    if p.exists():print(name,p.read_text()[-2000:])
p=Path('/home/ubuntu/joint-selection-v642-h100.tar.record.json')
if p.exists():print('archive_record',p.read_text())
