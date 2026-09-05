#!/usr/bin/env bash
# Native-Linux stand-in for the Windows `nvidia-smi.exe pmon -c 1` host-context channel of the G4
# contamination monitor. On WSL the monitor detected foreign compute through /dev/dxg holders plus the
# host pmon; on the H100 (native driver) there is no /dev/dxg and no host, but `nvidia-smi pmon` lists
# every compute context with its SM/memory utilisation. Rows belonging to the campaign worker itself
# (this script's parent) or its descendants (the --g4-session executor) are dropped so the worker
# never flags its own attempts; every other compute context is passed through verbatim and the
# monitor flags any with non-zero SM or memory utilisation (run-and-flag, amendment v1.2).
# The monitor calls: <this> pmon -c 1
set -uo pipefail
owner=${G4_MONITOR_OWNER_PID:-$PPID}
is_descendant_of_owner() {
  local pid=$1 guard=0
  while [ "$pid" -gt 1 ] && [ "$guard" -lt 64 ]; do
    [ "$pid" -eq "$owner" ] && return 0
    pid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    guard=$((guard + 1))
  done
  return 1
}
nvidia-smi pmon -c 1 2>/dev/null | while IFS= read -r line; do
  case "$line" in
    "#"*|"") echo "$line"; continue ;;
  esac
  pid=$(echo "$line" | awk '{print $2}')
  case "$pid" in ''|*[!0-9]*) echo "$line"; continue ;; esac
  if is_descendant_of_owner "$pid"; then
    continue
  fi
  echo "$line"
done
