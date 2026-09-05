#!/bin/bash
# Retain the failed G0 attempt (as the b6afb49 seal did under g0/failures/) and relaunch from G0.
set -euo pipefail
cd "$HOME/spacepdhcg/v1"
ev=results/gpu/current-head-9e75b47-h100
test "$(head -1 "$ev/g0/status.txt")" = "status=FAIL"
mkdir -p "$ev/g0/failures"
n=$(find "$ev/g0/failures" -maxdepth 1 -name 'attempt-*' | wc -l)
dest="$ev/g0/failures/attempt-$((n+1))"
mkdir -p "$dest"
shopt -s nullglob
for f in "$ev/g0"/*; do
  case "$(basename "$f")" in run.sh|failures) ;; *) mv "$f" "$dest/";; esac
done
printf 'reason=scikit-build-core could not find cmake>=3.24: the orchestrator PATH lacked the venv (system cmake 3.22); fixed by exporting PATH=.venv/bin first\n' > "$dest/reason.txt"
rm -rf build-current-head-g0-wheel-consumer
ls "$dest" | head -40
test -z "$(git status --porcelain=v1)"
cp /home/ubuntu/s/reseal_all.sh /home/ubuntu/s/reseal_all.sh.bak 2>/dev/null || true
SKIP_PREFLIGHT=1 nohup bash /home/ubuntu/s/reseal_all.sh >> /home/ubuntu/logs/reseal_all.sh.log 2>&1 < /dev/null &
echo "relaunched pid $!"
