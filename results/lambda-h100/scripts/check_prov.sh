#!/bin/bash
echo "== provision log tail"; tail -15 /home/ubuntu/logs/provision.sh.log
echo "== apt/dpkg running?"; pgrep -a apt-get | head -3; pgrep -a dpkg | head -3
echo "== would-install list (driver check)"; grep -ciE 'nvidia-driver|nvidia-kernel|nvidia-dkms|cuda-drivers|libnvidia-' /tmp/cuda-toolkit-12-8.would-install.txt; grep -iE 'nvidia|driver' /tmp/cuda-toolkit-12-8.would-install.txt | head
echo "== base tools present?"; for t in ninja jq tmux rsync gdb xz unzip; do printf '%s=%s ' "$t" "$(command -v $t || echo MISSING)"; done; echo
echo "== bundles"; for b in v1 v2 gtoc12 pdhcg qoco; do printf '%s: ' "$b"; git bundle verify /home/ubuntu/bundles/$b.bundle 2>&1 | tail -1; done
echo "== lambda mpi"; ls /usr/mpi/gcc/openmpi-4.1.7rc1/bin | head -5; /usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun --version 2>&1 | head -1
