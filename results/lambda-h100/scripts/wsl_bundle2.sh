set -eu
cd /home/angus/worktrees/spacepdhcg-gtoc12-methods
echo "HEAD=$(git rev-parse --short HEAD) branch=$(git branch --show-current) dirty=$(git status --porcelain=v1 | wc -l)"
git log --oneline -5
git status --porcelain=v1 | head
echo "== files in b0d5201..HEAD"; git diff --stat f81e834 HEAD | tail -15
echo "== tracked joint/fm7 files"; git ls-files results/gtoc12/runs/joint_itinerary_v2 | wc -l; git ls-files results/gtoc12/runs/fleet_master_v7 | wc -l
du -sh results/gtoc12/runs/joint_itinerary_v2 results/gtoc12/runs/fleet_master_v7
echo "== untracked/ignored under those runs"; git status --porcelain=v1 --ignored -- results/gtoc12/runs/joint_itinerary_v2 results/gtoc12/runs/fleet_master_v7 | head -8
out=/home/angus/bundles/to-h100/gtoc12-methods-8e15b92.bundle
git bundle create "$out" f81e834..feat/gtoc12-joint-itinerary
git bundle verify "$out"
ls -la "$out"; sha256sum "$out"
cp "$out" /mnt/c/Users/Angus/h100work/gtoc12-methods-8e15b92.bundle
echo "== docs 6.10 joint section"; grep -n "^### 6\.\|^## " docs/GTOC12_TRACK.md | tail -12