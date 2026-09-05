#!/bin/bash
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"
export CUDA_VISIBLE_DEVICES=""
export GIT_AUTHOR_NAME="SpacePDHCG-Integration" GIT_AUTHOR_EMAIL="integration@spacepdhcg.local"
export GIT_COMMITTER_NAME="SpacePDHCG-Integration" GIT_COMMITTER_EMAIL="integration@spacepdhcg.local"
cd "$HOME/spacepdhcg/gtoc12"
export PYTHONPATH="$PWD/src"
py=.venv/bin/python
log "HEAD $(git rev-parse --short HEAD) dirty=$(git status --porcelain=v1 | wc -l)"
$py "$HOME/s/patch_partitions.py" || exit 2
grep -n "Sequence" src/spacepdhcg/gtoc12/bundles.py | head -3
grep -n "^from .clusters import\|^from .search import\|SearchSettings" src/spacepdhcg/gtoc12/bundles.py | head -5
if $py -m ruff --version >/dev/null 2>&1; then
  $py -m ruff format src/spacepdhcg/gtoc12/bundles.py src/spacepdhcg/gtoc12/cli.py tests/test_gtoc12_bundles.py
  $py -m ruff check src/spacepdhcg/gtoc12 tests/test_gtoc12_bundles.py || exit 3
else
  log "ruff not in venv; trying uvx ruff"
  uvx ruff format src/spacepdhcg/gtoc12/bundles.py src/spacepdhcg/gtoc12/cli.py tests/test_gtoc12_bundles.py && uvx ruff check src/spacepdhcg/gtoc12 tests/test_gtoc12_bundles.py || exit 3
fi
log "== focused tests"
$py -m pytest -q -p no:cacheprovider tests/test_gtoc12_bundles.py -k "family_partitions or cluster_band_partitions or rank_families" 2>&1 | tail -5
$py -m spacepdhcg gtoc12 cluster-fleet --help | grep -A2 "cluster-radius\|all-family-bands" | head -12
git status --porcelain=v1