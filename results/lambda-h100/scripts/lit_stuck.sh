#!/bin/bash
pid=$(pgrep -f 'literature gpu-run' | head -1)
echo "pid=$pid state=$(cat /proc/$pid/status | grep -E '^State' ) wchan=$(cat /proc/$pid/wchan 2>/dev/null)"
echo "== children"; ps --ppid "$pid" -o pid,etime,time,pcpu,stat,args | cut -c1-200
for c in $(pgrep -P "$pid"); do echo "child $c: $(cat /proc/$c/wchan 2>/dev/null) threads=$(ls /proc/$c/task | wc -l)"; ps --ppid "$c" -o pid,etime,time,pcpu,stat,args | cut -c1-200; done
echo "== threads of main"; for t in /proc/$pid/task/*; do echo "$(basename $t) $(cat $t/stat | cut -d' ' -f3) $(cat $t/wchan 2>/dev/null)"; done | head -20
echo "== open files (tail)"; ls -la /proc/$pid/fd 2>/dev/null | grep -v 'pipe\|socket\|/dev/' | tail -8
echo "== chari runner source hints"; cd "$HOME/spacepdhcg/v2" && grep -rn 'def .*gpu_persistent_batch\|batch_sizes\|\[1, 16, 64\]\|subprocess' src/spacepdhcg/literature/*.py | head -12
