#!/bin/bash
source ~/spacepdhcg/env.sh
cd ~/spacepdhcg/v1
mkdir -p ~/manifests
.venv/bin/python scripts/gpu/verify_environment.py --output ~/manifests/environment-v1-h100.json; echo "verify_environment exit=$?"
python3 - <<'PY'
import json
d = json.load(open('/home/ubuntu/manifests/environment-v1-h100.json'))
print("top-level keys:", list(d))
def show(k):
    v = d.get(k)
    s = json.dumps(v, indent=1, sort_keys=True)
    print(f"--- {k}\n{s[:1800]}")
for k in list(d)[:12]:
    if isinstance(d[k], (dict, list)):
        show(k)
    else:
        print(k, "=", d[k])
PY
echo "== reseal log"; tail -5 ~/logs/reseal_all.sh.log
