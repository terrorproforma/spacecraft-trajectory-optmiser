#!/usr/bin/env bash
# Resume the H100 reseal at the seals stage in the correct order:
# summarize (writes all summary.json incl. current-head-summary.json) -> validate -> seal -> verify.
set -euo pipefail
source /home/ubuntu/spacepdhcg/env.sh
root=/home/ubuntu/spacepdhcg/v1
cd "$root"
export PATH="$root/.venv/bin:$PATH"
sha7=9e75b47
ev="results/gpu/current-head-${sha7}-h100"
logdir=/home/ubuntu/logs/reseal-${sha7}
statusf="$logdir/status.txt"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
step() {
  local name="$1"; shift
  echo "[$(ts)] START $name"; echo "status=RUNNING step=$name" > "$statusf"
  if "$@" > "$logdir/$name.log" 2>&1; then
    echo "[$(ts)] PASS  $name"
  else
    echo "[$(ts)] FAIL  $name -- see $logdir/$name.log"; echo "status=FAIL step=$name" > "$statusf"
    tail -20 "$logdir/$name.log"; exit 1
  fi
}
# Retain the mis-ordered first attempt's validation log for the record.
mv -f "$logdir/seals-validate.log" "$logdir/seals-validate.attempt1-misordered.log" 2>/dev/null || true
# The root evidence index must not include a log that archive_run.py itself writes while indexing:
# redirect root-index.log outside the evidence tree and drop the stale in-tree copy.
sed -i "s#>\"\${seals}/root-index.log\" 2>&1#>\"$logdir/root-index.log\" 2>\&1#" "$ev/seals/seal.sh"
grep -n 'root-index.log' "$ev/seals/seal.sh"
rm -f "$ev/seals/root-index.log" "$ev/evidence-index.json" "$ev/evidence-index.json.sha256"
rm -rf "$ev/seals/__pycache__"
step seals-summarize .venv/bin/python "$ev/seals/summarize.py"
step seals-validate bash "$ev/seals/validate.sh"
step seals-seal bash "$ev/seals/seal.sh"
step seals-verify .venv/bin/python "$ev/seals/verify_seals.py"
echo "status=PASS step=all" > "$statusf"
echo "[$(ts)] RESEAL COMPLETE"
cat "$ev/current-head-summary.json"
cat "$ev/evidence-index.json.sha256"
ls -la "$ev/seals/"*.tar.gz
