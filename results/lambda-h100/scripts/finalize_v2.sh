#!/bin/bash
# After gtoc12_v2_campaign.sh reports stage=done: docs, viewer export, commits, bundle, compact tarball.
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"; export CUDA_VISIBLE_DEVICES=""
export GIT_AUTHOR_NAME="SpacePDHCG-Integration" GIT_AUTHOR_EMAIL="integration@spacepdhcg.local"
export GIT_COMMITTER_NAME="SpacePDHCG-Integration" GIT_COMMITTER_EMAIL="integration@spacepdhcg.local"
cd "$HOME/spacepdhcg/gtoc12"; export PYTHONPATH="$PWD/src"; py=.venv/bin/python; runs=results/gtoc12/runs
grep -q "stage=done" "$HOME/logs/gtoc12-v2-RESULT" || { log "campaign not finished: $(cat $HOME/logs/gtoc12-v2-RESULT)"; exit 2; }
pgrep -f "spacepdhcg gtoc12" >/dev/null && { log "gtoc12 processes still running"; exit 2; }
log "HEAD $(git rev-parse --short HEAD) dirty=$(git status --porcelain=v1 | wc -l)"

add_run() {  # force-add the compact artefacts of one run (results/ is git-ignored)
  local r="$runs/$1"; [ -d "$r" ] || return 0
  find "$r" -maxdepth 1 \( -name run_report.json -o -name chain_stats.json -o -name 'official_verification*.json' -o -name independent_verify.txt -o -name ships.jsonl \) -print0 | xargs -0 -r git add -f
  find "$r" -path '*/clusters/*' \( -name bundle.json -o -name route_summary.json \) -print0 | xargs -0 -r git add -f
  find "$r" -path '*/ships/*' -name route_summary.json -print0 | xargs -0 -r git add -f
  find "$r" -path '*/fleets/*' -name fleet.json -print0 | xargs -0 -r git add -f
  [ -d "$r/fleet" ] && find "$r/fleet" \( -name Result.txt -o -name fleet.json -o -name manifest.json \) -print0 | xargs -0 -r git add -f
}

# -- commit 1: the H100 v1 campaign artefacts (never committed on this clone)
add_run cluster_fleet_h100_v1; add_run fleet_master_h100_v1
git diff --cached --quiet || git commit -q -m "gtoc12 results: cluster_fleet_h100_v1 (10699.5 kg, 19 ships) and fleet_master_h100_v1 (11517.6 kg, 20 ships, 163 asteroids) from the Lambda H100 host

16 workers on cores 10-25, 6 h budget, v6 recipe; archive-wide master over 13 sources
(1055 columns). Official GTOC12_Verify passes every fleet/candidate Result.txt."
log "commit 1: $(git rev-parse --short HEAD)"

# -- viewer export of the new best fleet (+ the v2 importer, output ignored)
best=$runs/fleet_master_h100_v2/fleet/Result.txt
if [ -f "$best" ]; then
  log "== export-viewer"; $py -m spacepdhcg gtoc12 export-viewer "$best" --output results/gtoc12/viewer-exports/fleet_master_h100_v2 --run-id fleet_master_h100_v2_fleet > "$HOME/logs/gtoc12-export_viewer_h100_v2.log" 2>&1; tail -2 "$HOME/logs/gtoc12-export_viewer_h100_v2.log"
  if [ -d "$HOME/spacepdhcg/v2/web/trajectory-viewer" ]; then
    ( cd "$HOME/spacepdhcg/v2/web/trajectory-viewer" && npm run import-gtoc12 -- --export "$HOME/spacepdhcg/gtoc12/results/gtoc12/viewer-exports/fleet_master_h100_v2" --catalogue "$HOME/spacepdhcg/gtoc12/benchmarks/gtoc12/data/GTOC12_Asteroids_Data.txt" --solution "$HOME/spacepdhcg/gtoc12/$best" --fleet "$HOME/spacepdhcg/gtoc12/$runs/fleet_master_h100_v2/fleet/fleet.json" ) > "$HOME/logs/gtoc12-viewer_import_h100_v2.log" 2>&1; tail -3 "$HOME/logs/gtoc12-viewer_import_h100_v2.log"
  fi
fi

# -- commit 2: v2 campaign artefacts + docs
$py "$HOME/s/docs_v2.py" || log "docs update skipped"
for r in cluster_fleet_h100_v2 joint_itinerary_h100_v1 joint_itinerary_h100_v2 fleet_master_h100_v2 cluster_fleet_v8; do add_run $r; done
[ -f results/gtoc12/leg_stats/after_h100_v2.json ] && git add -f results/gtoc12/leg_stats/after_h100_v2.json
git add docs/GTOC12_TRACK.md
summary=$($py - <<'PY'
import json
fm=json.load(open("results/gtoc12/runs/fleet_master_h100_v2/run_report.json")); b=fm["best"]; m=fm["master"]
cf=json.load(open("results/gtoc12/runs/cluster_fleet_h100_v2/run_report.json"))
print(f"fleet_master_h100_v2 {b['score_kg']:.2f} kg, {b['fleet']['ships']} ships, {len(b['fleet']['asteroids'])} asteroids, {b['fleet']['average_collected_kg']:.2f} kg avg, LP gap {m.get('lp_gap_kg',0):.1f} kg; cluster_fleet_h100_v2 {cf['best']['score_kg']:.1f} kg / {cf['best']['fleet']['ships']} ships from {len(cf['bundles'])} families")
PY
)
git diff --cached --quiet || git commit -q -m "gtoc12 results: H100 v2 campaign - $summary

cluster_fleet_h100_v2 (22 workers, 8 h, union of four family partitions), joint_itinerary_h100_v1/v2
(whole-itinerary joint re-optimisation of every archived chain >= 450 kg), fleet_master_h100_v2
(archive-wide master with the LP bound), official + independent verification, chain-mass
distribution, leg stats, docs section 7."
log "commit 2: $(git rev-parse --short HEAD)"; git log --oneline -4

# -- bundle back to WSL (fetch-only refs) and a compact tarball for the Windows results tree
sha=$(git rev-parse --short HEAD); mkdir -p "$HOME/bundles/from-h100" "$HOME/stage"
git bundle create "$HOME/bundles/from-h100/gtoc12-h100-v2-$sha.bundle" c4e2c31..feat/gtoc12-asteroid-mining && git bundle verify "$HOME/bundles/from-h100/gtoc12-h100-v2-$sha.bundle" | tail -1
tar czf "$HOME/stage/gtoc12-h100-v2-compact.tgz" \
  --exclude='clusters/*/ship_*/Result.txt' --exclude='fleets/*/Result.txt' --exclude='columns' --exclude='trajectories.json' --exclude='ships/*/*/Result.txt' \
  $runs/cluster_fleet_h100_v2 $runs/joint_itinerary_h100_v1 $runs/joint_itinerary_h100_v2 $runs/fleet_master_h100_v2 results/gtoc12/leg_stats/after_h100_v2.json \
  -C "$HOME" logs/gtoc12-cluster_fleet_h100_v2.log logs/gtoc12-joint_itinerary_h100_v1.log logs/gtoc12-joint_itinerary_h100_v2.log logs/gtoc12-fleet_master_h100_v2.log logs/gtoc12-verify_h100_v2.log logs/gtoc12_v2_campaign.sh.log logs/gtoc12-v2-RESULT 2>/dev/null
ls -la "$HOME/stage/gtoc12-h100-v2-compact.tgz" "$HOME/bundles/from-h100/gtoc12-h100-v2-$sha.bundle"
echo "$sha" > "$HOME/stage/gtoc12-h100-v2-HEAD"
log "done: HEAD $sha"