#!/bin/bash
# Merge the WSL gtoc12 branches (7d2e301 asteroid-mining, f81e834 joint-itinerary) into the H100 clone at c4e2c31.
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"
export CUDA_VISIBLE_DEVICES=""
export GIT_AUTHOR_NAME="SpacePDHCG-Integration" GIT_AUTHOR_EMAIL="integration@spacepdhcg.local"
export GIT_COMMITTER_NAME="SpacePDHCG-Integration" GIT_COMMITTER_EMAIL="integration@spacepdhcg.local"
cd "$HOME/spacepdhcg/gtoc12"
log "HEAD $(git rev-parse --short HEAD) dirty=$(git status --porcelain=v1 | wc -l)"
[ "$(git status --porcelain=v1 | wc -l)" -eq 0 ] || { log "tree dirty, abort"; git status --porcelain=v1 | head; exit 2; }
[ "$(git rev-parse --short HEAD)" = "c4e2c31" ] || { log "unexpected HEAD"; exit 2; }
bundle="$HOME/bundles/gtoc12-wsl-7d2e301-f81e834.bundle"
sha256sum "$bundle"
git bundle verify "$bundle" || exit 2
git fetch "$bundle" refs/heads/feat/gtoc12-asteroid-mining:refs/wsl/gtoc12-asteroid-mining refs/heads/feat/gtoc12-joint-itinerary:refs/wsl/gtoc12-joint-itinerary || exit 2
log "fetched: $(git rev-parse --short refs/wsl/gtoc12-asteroid-mining) $(git rev-parse --short refs/wsl/gtoc12-joint-itinerary)"
grep -q "test_gtoc12_cooperative.py merge=union" .git/info/attributes 2>/dev/null || echo "tests/test_gtoc12_cooperative.py merge=union" >> .git/info/attributes
log "== merge 1: feat/gtoc12-asteroid-mining 7d2e301"
git merge --no-ff --no-edit -m "Merge WSL feat/gtoc12-asteroid-mining 7d2e301 (v7/v8 campaign, harvest substitution, sweep cells, external archives) into the H100 gtoc12 line" refs/wsl/gtoc12-asteroid-mining
rc=$?
if [ $rc -ne 0 ]; then
  log "merge 1 conflicts:"; git diff --name-only --diff-filter=U
  conflicts=$(git diff --name-only --diff-filter=U)
  [ "$conflicts" = "src/spacepdhcg/gtoc12/cooperative.py" ] || { log "unexpected conflict set"; exit 3; }
  .venv/bin/python "$HOME/s/resolve_coop.py" || exit 3
  git add src/spacepdhcg/gtoc12/cooperative.py
  git commit --no-edit -m "Merge WSL feat/gtoc12-asteroid-mining 7d2e301 (v7/v8 campaign, harvest substitution, sweep cells, external archives) into the H100 gtoc12 line

Conflict: both lines raised the fleet-master recursion limit (ba9b764 columns+500,
c4e2c31 2*columns+200); resolved as the maximum of both margins. The two regression
tests are kept (union merge of tests/test_gtoc12_cooperative.py)." || exit 3
fi
log "merge 1 done: $(git rev-parse --short HEAD)"
log "== merge 2: feat/gtoc12-joint-itinerary f81e834"
git merge --no-ff --no-edit -m "Merge WSL feat/gtoc12-joint-itinerary f81e834 (whole-itinerary joint re-optimisation) into the H100 gtoc12 line" refs/wsl/gtoc12-joint-itinerary
rc=$?
if [ $rc -ne 0 ]; then log "merge 2 conflicts:"; git diff --name-only --diff-filter=U; git diff | head -80; exit 4; fi
log "merge 2 done: $(git rev-parse --short HEAD)"
git log --oneline --graph -8
grep -n "<<<<<<<\|>>>>>>>" src/spacepdhcg/gtoc12/cooperative.py tests/test_gtoc12_cooperative.py && exit 5
log "== quick import + recursion tests"
export PYTHONPATH="$PWD/src"
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_gtoc12_cooperative.py -k "recursion" 2>&1 | tail -5
log "done"