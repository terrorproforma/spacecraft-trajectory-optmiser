from pathlib import Path
import hashlib,json
repo=Path(__file__).resolve().parents[2]
source=repo/'results/lambda/2026-09-09/gpu-shared-return-options-v846/saved-audit.json'
prior=json.loads(source.read_text())
reference=dict(source=str(source.relative_to(repo)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    source_commit='2771155e469adb0590a4726438c06d84aff7f650',
    candidate_sha256={side:{ship:entry['plans_sha256'] for ship,entry in prior[side]['comparison'].items()} for side in ('local','h100')})
(repo/'results/lambda/2026-09-09/gpu-native-collection-plan-v859/reference.json').write_text(json.dumps(reference,indent=2)+'\n',newline='\n')
