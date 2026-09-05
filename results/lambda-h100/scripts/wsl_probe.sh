set -u
for w in /home/angus/worktrees/spacepdhcg-gtoc12 /home/angus/worktrees/spacepdhcg-gtoc12-methods; do
  echo "== $w"
  if [ -d "$w" ]; then
    cd "$w" || continue
    echo "HEAD=$(git rev-parse --short HEAD) branch=$(git branch --show-current) dirty=$(git status --porcelain=v1 | wc -l)"
    git log --oneline -6
    echo "-- dirty files:"; git status --porcelain=v1 | head -20
  else echo "missing"; fi
done
cd /home/angus/worktrees/spacepdhcg-gtoc12
echo "== c4e2c31 parent: $(git log --oneline -1 c4e2c31^)"
echo "== merge-base 7d2e301 c4e2c31: $(git merge-base HEAD refs/h100/gtoc12-asteroid-mining)"
echo "== files changed HEAD vs base"; git diff --stat $(git merge-base HEAD refs/h100/gtoc12-asteroid-mining) HEAD | tail -40
echo "== c4e2c31 patch"; git show --stat refs/h100/gtoc12-asteroid-mining | head -20
echo "== ba9b764 patch"; git show --stat ba9b764 | head -20
echo "== all branches with gtoc12"; git branch -a | grep -i gtoc
echo "== methods branch"; git log --oneline -5 feat/gtoc12-joint-itinerary 2>/dev/null || echo "no local branch feat/gtoc12-joint-itinerary"
git worktree list