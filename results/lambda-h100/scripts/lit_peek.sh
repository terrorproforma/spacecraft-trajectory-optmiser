#!/bin/bash
cd "$HOME/spacepdhcg/v2" || exit 1
ps -o pid,etime,time,pcpu,rss,args -p "$(pgrep -f 'literature gpu-run' | head -1)" 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "== expected per manifest"; python3 -c "
import json; m=json.load(open('benchmarks/gpu_deferred_validation_v2.json'))
for it in m['items']:
    if it['id']=='literature-gpu-run':
        print(json.dumps(it, indent=1)[:3000])
"
echo "== recent files under results/literature"; find results/literature -newer results/gpu/h100-deferred-3373988/items.tsv -type f | head; find /tmp -maxdepth 2 -newer results/gpu/h100-deferred-3373988/items.tsv -name '*literature*' 2>/dev/null | head
