ls ~/spacepdhcg/v2/web/trajectory-viewer/scripts/ ~/spacepdhcg/v2/web/trajectory-viewer/data/ 2>&1 | head -30
grep -n '"scripts"' -A 12 ~/spacepdhcg/v2/web/trajectory-viewer/package.json
cd ~/spacepdhcg/v2 && git rev-parse --short HEAD && git status --porcelain=v1 | wc -l
grep -n -i 'import\|gtoc12' ~/spacepdhcg/v2/web/trajectory-viewer/README.md | head -20
ls ~/spacepdhcg/gtoc12/results/gtoc12/runs/fleet_master_v7/fleet/viewer/