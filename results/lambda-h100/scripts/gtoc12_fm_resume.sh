#!/bin/bash
# Resume the GTOC12 H100 campaign at the fleet-master stage on the fixed source (c4e2c31), then verify.
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root="$HOME/spacepdhcg/gtoc12"
cd "$root"
export PYTHONPATH="$root/src"
py="$root/.venv/bin/python"
runs=results/gtoc12/runs
log "source $(git rev-parse HEAD) branch $(git branch --show-current) dirty=$(git status --porcelain=v1 | wc -l); affinity=$(taskset -pc $$ | cut -d: -f2)"
# Retain the failed first attempt (RecursionError at c495dc0) beside the new run.
if [ -d "$runs/fleet_master_h100_v1" ] && [ ! -d "$runs/fleet_master_h100_v1.attempt1-c495dc0-recursionerror" ]; then
  mv "$runs/fleet_master_h100_v1" "$runs/fleet_master_h100_v1.attempt1-c495dc0-recursionerror"
  cp "$HOME/logs/gtoc12-fleet_master_h100_v1.log" "$runs/fleet_master_h100_v1.attempt1-c495dc0-recursionerror/fleet-master.log"
fi
echo "status=RUNNING stage=fleet-master (resume on c4e2c31)" > "$HOME/logs/gtoc12-RESULT"
log "== fleet-master fleet_master_h100_v1 (archive-wide: 12 archived sources + cluster_fleet_h100_v1, 16 workers)"
"$py" -m spacepdhcg gtoc12 fleet-master \
  --run-id fleet_master_h100_v1 --output "$runs/fleet_master_h100_v1" \
  --source $runs/cluster_fleet_v1 --source $runs/cluster_fleet_v2_deep --source $runs/cluster_fleet_v3_repair \
  --source $runs/cluster_fleet_v4 --source $runs/fleet10_master_v1 --source $runs/probe_v4_family \
  --source $runs/cluster_fleet_v5 --source $runs/cluster_fleet_v5c --source $runs/probe_v5_family247 \
  --source $runs/cluster_fleet_v6 --source $runs/probe_v6_family --source $runs/return_sweep_v1 \
  --source $runs/cluster_fleet_h100_v1 \
  --workers 16 \
  > "$HOME/logs/gtoc12-fleet_master_h100_v1.log" 2>&1
rc=$?
log "fleet-master exit=$rc"
[ "$rc" -eq 0 ] || { echo "status=FAIL stage=fleet-master exit=$rc" > "$HOME/logs/gtoc12-RESULT"; tail -30 "$HOME/logs/gtoc12-fleet_master_h100_v1.log"; exit 1; }
"$py" - <<'PY'
import json
d = json.load(open("results/gtoc12/runs/fleet_master_h100_v1/run_report.json"))
print("status:", d.get("status"), "columns:", d.get("columns"), "recert_wall_s:", d.get("recertification_wall_seconds"), "master_wall_s:", d.get("master_wall_seconds"))
print("master:", json.dumps(d.get("master"))[:900])
print("score:", d.get("score_kg"), d.get("official", d.get("verification")))
PY

echo "status=RUNNING stage=official-verify" > "$HOME/logs/gtoc12-RESULT"
log "== official GTOC12_Verify over every emitted Result.txt in the two new runs"
"$py" - <<'PY'
import json, pathlib
from spacepdhcg.gtoc12.official import run_official_verifier
rows = []
for run in ("cluster_fleet_h100_v1", "fleet_master_h100_v1"):
    for path in sorted(pathlib.Path("results/gtoc12/runs", run).rglob("Result.txt")):
        s = run_official_verifier(path).summary()
        rows.append({"path": str(path), **s})
ok = sum(r["ok"] for r in rows)
print(f"official verifier: {ok}/{len(rows)} Result.txt files pass")
bad = [r for r in rows if not r["ok"]]
for r in bad[:20]:
    print("FAIL", r)
pathlib.Path("results/gtoc12/runs/fleet_master_h100_v1/official_verification.json").write_text(json.dumps({"passed": ok, "total": len(rows), "rows": rows}, indent=1))
fleet = [r for r in rows if r["path"].endswith("fleet_master_h100_v1/fleet/Result.txt") or r["path"].endswith("cluster_fleet_h100_v1/fleet/Result.txt")]
print("fleets:", json.dumps(fleet, indent=1))
raise SystemExit(0 if ok == len(rows) else 1)
PY
rc=$?
echo "status=$([ $rc -eq 0 ] && echo PASS || echo FAIL) stage=done verify_exit=$rc" > "$HOME/logs/gtoc12-RESULT"
log "== done (verify exit=$rc)"
