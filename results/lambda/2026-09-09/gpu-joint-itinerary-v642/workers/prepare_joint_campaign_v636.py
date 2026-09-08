from pathlib import Path
import hashlib,json,shutil
source=Path('/home/angus/spacepdhcg-joint-selection-v630/repo')
names=['build/performance/orphan-recovery-v595/run.py','results/gtoc12/hop_inflation_fit.json','results/lambda/2026-09-06/fleet_master_v11/fleet/Result.txt']
for name in names:
    path=source/name;assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,path)
(source.parent/'campaign-fixtures.json').write_text(json.dumps({name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names},indent=2))
