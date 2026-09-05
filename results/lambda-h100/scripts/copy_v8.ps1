. C:\Users\Angus\h100work\h.ps1
wrun wsl_pack_v8.sh
if ($LASTEXITCODE -ne 0) { Write-Host "cluster_fleet_v8 not finished yet"; exit 1 }
rput "C:\Users\Angus\h100work\cluster_fleet_v8.tgz" "stage/cluster_fleet_v8.tgz"
rsh "cd ~/spacepdhcg/gtoc12/results/gtoc12/runs && tar xzf ~/stage/cluster_fleet_v8.tgz && find cluster_fleet_v8 -name route_summary.json | wc -l && ls cluster_fleet_v8"