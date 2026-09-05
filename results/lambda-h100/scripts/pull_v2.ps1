# Pull the finalised H100 v2 campaign home: compact tarball -> results/lambda-h100/gtoc12, bundle -> WSL refs/h100/gtoc12-asteroid-mining
. C:\Users\Angus\h100work\h.ps1
$dest = "C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda-h100\gtoc12"
New-Item -ItemType Directory -Force -Path $dest, "C:\Users\Angus\h100work\pull" | Out-Null
rget "stage/gtoc12-h100-v2-compact.tgz" "C:\Users\Angus\h100work\pull\gtoc12-h100-v2-compact.tgz"
rget "stage/gtoc12-h100-v2-HEAD" "C:\Users\Angus\h100work\pull\HEAD"
$sha = (Get-Content "C:\Users\Angus\h100work\pull\HEAD").Trim()
rget "bundles/from-h100/gtoc12-h100-v2-$sha.bundle" "C:\Users\Angus\h100work\pull\gtoc12-h100-v2-$sha.bundle"
tar -xzf "C:\Users\Angus\h100work\pull\gtoc12-h100-v2-compact.tgz" -C "C:\Users\Angus\h100work\pull"
Copy-Item -Recurse -Force "C:\Users\Angus\h100work\pull\results\gtoc12\runs\*" $dest
Copy-Item -Force "C:\Users\Angus\h100work\pull\results\gtoc12\leg_stats\after_h100_v2.json" $dest
New-Item -ItemType Directory -Force -Path "$dest\..\logs" | Out-Null
Copy-Item -Force "C:\Users\Angus\h100work\pull\logs\*" "$dest\..\logs\"
Copy-Item -Force "C:\Users\Angus\h100work\pull\gtoc12-h100-v2-$sha.bundle" "$dest\..\bundles\"
wlf "C:\Users\Angus\h100work\s\wsl_fetch_v2.sh" @"
set -eu
mkdir -p /home/angus/bundles/from-h100
cp /mnt/c/Users/Angus/h100work/pull/gtoc12-h100-v2-$sha.bundle /home/angus/bundles/from-h100/
cd /home/angus/worktrees/spacepdhcg-gtoc12
git bundle verify /home/angus/bundles/from-h100/gtoc12-h100-v2-$sha.bundle | tail -1
git fetch /home/angus/bundles/from-h100/gtoc12-h100-v2-$sha.bundle +refs/heads/feat/gtoc12-asteroid-mining:refs/h100/gtoc12-asteroid-mining
git log --oneline -6 refs/h100/gtoc12-asteroid-mining
echo "ahead/behind vs local branch: `$(git rev-list --left-right --count refs/h100/gtoc12-asteroid-mining...feat/gtoc12-asteroid-mining)"
"@
wrun wsl_fetch_v2.sh
Get-ChildItem $dest | Select-Object Name