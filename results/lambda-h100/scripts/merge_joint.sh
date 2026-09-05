#!/bin/bash
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"; export CUDA_VISIBLE_DEVICES=""
export GIT_AUTHOR_NAME="SpacePDHCG-Integration" GIT_AUTHOR_EMAIL="integration@spacepdhcg.local"
export GIT_COMMITTER_NAME="SpacePDHCG-Integration" GIT_COMMITTER_EMAIL="integration@spacepdhcg.local"
cd "$HOME/spacepdhcg/gtoc12"; export PYTHONPATH="$PWD/src"
log "tests log tail:"; tail -3 "$HOME/logs/gtoc12-tests-merged.log"; pgrep -f "pytest -q -p no:cacheprovider tests/test_gtoc12" >/dev/null && log "tests still running"
log "== commit the partition patch"
git add src/spacepdhcg/gtoc12/bundles.py src/spacepdhcg/gtoc12/cli.py tests/test_gtoc12_bundles.py
git commit -q -m "gtoc12 cluster-fleet: price a union of family partitions (several radii x band sets)

The H100 v1 campaign exhausted its 47 collect-window families (radius 1.75, >= 20
members) after 2.5 h of a 6 h budget, every family cut at 2-3 of 5 ships by the
per-family budget. bundles.family_partitions clusters and ranks each (radius, band
set) partition on its own visit epochs, offsets the labels per partition
(FAMILY_LABEL_STRIDE) so clusters/family_* directories and column identifiers never
collide, drops duplicate member sets and returns one cheapest-first list.
--cluster-radius accepts a comma-separated list and --all-family-bands adds the
phasing-aware deploy/collect partition next to the collect-window one; the run report
lists the partitions and tags every priced family. On the 10612-asteroid pool at
>= 20 members: 47 + 56 + 29 + 35 = 167 unique families for radii 1.75,1.6 (ranking
203 s). The budget marks gain the 480-minute (8 h) mark." || exit 2
log "committed $(git rev-parse --short HEAD)"
bundle="$HOME/bundles/gtoc12-methods-8e15b92.bundle"
sha256sum "$bundle"; git bundle verify "$bundle" || exit 2
git fetch "$bundle" refs/heads/feat/gtoc12-joint-itinerary:refs/wsl/gtoc12-joint-itinerary || exit 2
log "fetched $(git rev-parse --short refs/wsl/gtoc12-joint-itinerary)"
git merge --no-ff --no-edit -m "Merge WSL feat/gtoc12-joint-itinerary 8e15b92 (joint_itinerary_v1/v2, fleet_master_v7 = 21 ships 12346.48 kg, memory entries) into the H100 gtoc12 line" refs/wsl/gtoc12-joint-itinerary
rc=$?
if [ $rc -ne 0 ]; then log "conflicts:"; git diff --name-only --diff-filter=U; git diff | head -120; exit 4; fi
log "merged: $(git rev-parse --short HEAD)"; git log --oneline -4
ls results/gtoc12/runs/ | tr '\n' ' '; echo
ls results/gtoc12/runs/joint_itinerary_v2 results/gtoc12/runs/fleet_master_v7
python3 - <<'PY'
import json
d=json.load(open("results/gtoc12/runs/fleet_master_v7/fleet/fleet.json"))
print("fm7 score", d.get("score_kg"), "ships", d["fleet"]["ships"], "asteroids", len(d["fleet"]["asteroids"]), "avg", d["fleet"]["average_collected_kg"])
PY
.venv/bin/python -m spacepdhcg gtoc12 joint-itinerary --help