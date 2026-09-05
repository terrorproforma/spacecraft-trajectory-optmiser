#!/bin/bash
pid=$(pgrep -f 'literature gpu-run' | head -1)
echo "gpu-run pid=${pid:-none}"
if [ -n "$pid" ]; then
  ps -o pid,etime,time,pcpu,stat,args -p "$pid" | cut -c1-140
  echo "state: $(grep State /proc/$pid/status) wchan=$(cat /proc/$pid/wchan)"
  echo "children: $(pgrep -P "$pid" | wc -l)"; ps --ppid "$pid" -o pid,etime,time,pcpu,stat | head -5
  echo "threads:"; for t in /proc/$pid/task/*; do echo " $(basename $t) $(cut -d' ' -f3 $t/stat) $(cat $t/wchan 2>/dev/null)"; done | head -8
  echo "py-spy?"; which py-spy || true
  echo "gdb bt of main thread (python frames unavailable) ->  skip"
fi
echo "== rerun log"; cat "$HOME/logs/lit_rerun.sh.log"
echo "== blackmore gpu status now"; python3 - <<'PY'
import json
d=json.load(open('/home/ubuntu/spacepdhcg/v2/results/literature/blackmore-2010-pd3-case1.json'))
m=d['measured']; print({k:v for k,v in m.items() if 'gpu' in k})
PY
