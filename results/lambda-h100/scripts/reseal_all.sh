#!/bin/bash
# H100 current-head reseal orchestrator: preflight -> G0 -> G1 -> G2 -> G3 -> validate/summarize/seal/verify.
set -uo pipefail
source "$HOME/spacepdhcg/env.sh"
root="$HOME/spacepdhcg/v1"
cd "$root"
# The venv's cmake 4.4.3/ctest/ruff must be first on PATH: scikit-build-core (package-build, G1 pdhcg
# install) searches PATH for cmake>=3.24 and the system cmake is 3.22.
export PATH="$root/.venv/bin:$PATH"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
head=$(git rev-parse HEAD)
ev="results/gpu/current-head-${head:0:7}-h100"
logdir="$HOME/logs/reseal-${head:0:7}"
mkdir -p "$logdir" "$ev/preflight" "$ev/g0" "$ev/g1" "$ev/g2" "$ev/g3" "$ev/seals"
stamp() { date -u +%FT%TZ; }
step() { # name command...
  local name=$1; shift
  echo "[$(stamp)] START $name"
  if "$@" > "$logdir/$name.log" 2>&1; then
    echo "[$(stamp)] PASS  $name"
  else
    echo "[$(stamp)] FAIL  $name (exit $?) -- see $logdir/$name.log"; tail -30 "$logdir/$name.log"; echo "status=FAIL step=$name" > "$logdir/RESULT"; exit 1
  fi
}
echo "status=RUNNING" > "$logdir/RESULT"
test -z "$(git status --porcelain=v1)" || { echo "dirty tree"; exit 2; }
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo "GPU busy"; exit 3; }
if [ "$SKIP_PREFLIGHT" != "1" ]; then
  step preflight-capture bash "$ev/preflight/capture.sh"
  step preflight-build-qoco bash "$ev/preflight/build_qoco.sh"
else
  echo "[$(stamp)] preflight already sealed (status: $(head -1 "$ev/preflight/status.txt"), qoco: $(cat "$ev/preflight/qoco-build.status"))"
fi
step g0 bash "$ev/g0/run.sh"
step g1 bash "$ev/g1/run.sh"
step g2 bash "$ev/g2/run.sh"
step g3 bash "$ev/g3/run.sh"
step seals-validate bash "$ev/seals/validate.sh"
step seals-summarize .venv/bin/python "$ev/seals/summarize.py"
step seals-seal bash "$ev/seals/seal.sh"
step seals-verify .venv/bin/python "$ev/seals/verify_seals.py"
echo "status=PASS" > "$logdir/RESULT"
echo "[$(stamp)] RESEAL COMPLETE: $ev"
cat "$ev/current-head-summary.json"
