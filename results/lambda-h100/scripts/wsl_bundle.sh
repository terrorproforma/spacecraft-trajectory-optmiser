set -eu
cd /home/angus/worktrees/spacepdhcg-gtoc12
mkdir -p /home/angus/bundles/to-h100
out=/home/angus/bundles/to-h100/gtoc12-wsl-7d2e301-f81e834.bundle
git bundle create "$out" c495dc0..feat/gtoc12-asteroid-mining c495dc0..feat/gtoc12-joint-itinerary
git bundle verify "$out" || true
ls -la "$out"
sha256sum "$out"
cp "$out" /mnt/c/Users/Angus/h100work/gtoc12-wsl-7d2e301-f81e834.bundle
echo copied