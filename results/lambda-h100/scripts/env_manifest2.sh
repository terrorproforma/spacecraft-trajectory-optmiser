#!/bin/bash
source ~/spacepdhcg/env.sh
for wt in v1 v2; do
  cd ~/spacepdhcg/$wt
  PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/gpu/verify_environment.py --output ~/manifests/environment-$wt-h100.json; echo "$wt verify_environment exit=$?"
done
python3 - <<'PY'
import json
d = json.load(open('/home/ubuntu/manifests/environment-v1-h100.json'))
print(json.dumps(d["validation"], indent=1)[:2500])
c = d["commands"]
for k in ("cmake", "nvcc", "nvidia_smi", "ninja", "mpirun", "mpicc", "node"):
    if k in c:
        print(k, "->", (c[k]["stdout"] or c[k]["stderr"]).splitlines()[:1])
print("keys:", list(c))
PY
echo "== reseal log"; tail -4 ~/logs/reseal_all.sh.log
