#!/bin/bash
# Stage compact evidence for transfer to the Windows workspace (results/lambda-h100/).
set -uo pipefail
home="$HOME/evidence-home"
rm -rf "$home"; mkdir -p "$home/manifests" "$home/logs" "$home/bundles" "$home/gtoc12" "$home/g4"
cp ~/manifests/*.json "$home/manifests/" 2>/dev/null
cp ~/spacepdhcg/env.sh "$home/manifests/env.sh"
cp ~/logs/*.log ~/logs/gtoc12-RESULT "$home/logs/" 2>/dev/null
cp -r ~/logs/reseal-* "$home/logs/" 2>/dev/null
cp ~/s/*.sh ~/s/*.py "$home/logs/" 2>/dev/null; mkdir -p "$home/scripts"; mv "$home"/logs/*.sh "$home"/logs/*.py "$home/scripts/" 2>/dev/null

v1=~/spacepdhcg/v1
ev=$(ls -d $v1/results/gpu/current-head-*-h100 2>/dev/null | head -1)
if [ -n "$ev" ]; then
  # Keep the seal archives (small) and every index/sha256; drop the wheel/sdist and Nsight report binaries.
  rsync -a --exclude '*.nsys-rep' --exclude '*.sqlite' --exclude 'g0/artifacts/*.whl' --exclude 'g0/artifacts/*.tar.gz' \
    "$ev/" "$home/v1-reseal-$(basename "$ev" | sed 's/current-head-//')/"
fi
v2=~/spacepdhcg/v2
dv=$(ls -d $v2/results/gpu/h100-deferred-* 2>/dev/null | head -1)
[ -n "$dv" ] && rsync -a --exclude 'trajectories.json' "$dv/" "$home/v2-deferred-$(basename "$dv" | sed 's/h100-deferred-//')/"
# The literature gpu-run rewrote tracked report twins; carry the diff home for review (not committed here).
git -C "$v2" diff -- docs/REFERENCE_REPRODUCTION_REPORT.md benchmarks/literature/reference_reproduction.json results/literature \
  > "$home/v2-literature-report-twins.patch"

g=~/spacepdhcg/gtoc12/results/gtoc12/runs
for run in cluster_fleet_h100_v1 fleet_master_h100_v1 fleet_master_h100_v1.attempt1-c495dc0-recursionerror; do
  if [ -d "$g/$run" ]; then
    # run reports, verification, fleet solutions and candidate fleets; skip the 900+ per-ship diagnostic files
    rsync -a --include '*/' --include 'run_report.json' --include 'official_verification*.json' --include 'fleet-master.log' \
      --include '/fleet/**' --include '/fleets/**' --exclude '*' --prune-empty-dirs "$g/$run/" "$home/gtoc12/$run/"
  fi
done
cp -r "$g/cluster_fleet_h100_v1/fleet/viewer" "$home/gtoc12/cluster_fleet_h100_v1/fleet/" 2>/dev/null || true
cd "$HOME/spacepdhcg/gtoc12"; base=$(git rev-parse bundle/feat/gtoc12-asteroid-mining); head=$(git rev-parse HEAD)
if [ "$base" != "$head" ]; then
  git bundle create "$home/bundles/gtoc12-h100-fixes-${head:0:7}.bundle" "$base..feat/gtoc12-asteroid-mining" 2>/dev/null
  git log --oneline "$base..HEAD" > "$home/bundles/gtoc12-h100-fixes-${head:0:7}.log"
  git bundle verify "$home/bundles/gtoc12-h100-fixes-${head:0:7}.bundle" 2>&1 | tail -1
fi
cp ~/logs/gtoc12-*.log "$home/gtoc12/" 2>/dev/null

# G4 status on this host.
cp ~/g4/*.txt ~/g4/*.json "$home/g4/" 2>/dev/null

# Source fixes made on the instance: bundle every commit not in the bundles we received.
cd $v1
base=$(git rev-parse bundle/integration/single-gpu-v1); head=$(git rev-parse HEAD)
if [ "$base" != "$head" ]; then
  git bundle create "$home/bundles/single-gpu-v1-h100-fixes-${head:0:7}.bundle" "$base..integration/single-gpu-v1" 2>/dev/null
  git log --oneline "$base..HEAD" > "$home/bundles/single-gpu-v1-h100-fixes-${head:0:7}.log"
  git bundle verify "$home/bundles/single-gpu-v1-h100-fixes-${head:0:7}.bundle" 2>&1 | tail -1
fi
cd $v2
base=$(git rev-parse bundle/integration/single-gpu-v2-candidate); head=$(git rev-parse HEAD)
if [ "$base" != "$head" ]; then
  git bundle create "$home/bundles/single-gpu-v2-h100-fixes-${head:0:7}.bundle" "$base..integration/single-gpu-v2-candidate" 2>/dev/null
  git log --oneline "$base..HEAD" > "$home/bundles/single-gpu-v2-h100-fixes-${head:0:7}.log"
  git bundle verify "$home/bundles/single-gpu-v2-h100-fixes-${head:0:7}.bundle" 2>&1 | tail -1
fi
for w in v1 v2 gtoc12; do
  { echo "HEAD $(git -C ~/spacepdhcg/$w rev-parse HEAD) $(git -C ~/spacepdhcg/$w branch --show-current)"; git -C ~/spacepdhcg/$w status --porcelain=v1; } > "$home/worktree-status-$w.txt"
done
date -u +%FT%TZ > "$home/STAGED_UTC.txt"
du -sh "$home"; find "$home" -type f | wc -l
