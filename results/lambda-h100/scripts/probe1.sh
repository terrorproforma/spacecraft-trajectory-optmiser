#!/bin/bash
echo "== dpkg"
dpkg -l | grep -iE 'openmpi|nccl|ninja|build-essential|jq|cmake|libcudnn|cudss|cuda-toolkit|nvidia-container' | awk '{print $2, $3}'
echo "== apt sources"
ls /etc/apt/sources.list.d/
grep -h -v '^#' /etc/apt/sources.list.d/* 2>/dev/null | grep -v '^$' | head -20
echo "== cuda dirs"
ls /usr/local/ | grep -i cuda
ls /usr/local/cuda/lib64 | grep -iE 'cudss|nccl|cusparse|cublas' | head
ls /usr/lib/x86_64-linux-gnu | grep -iE 'nccl|cudss|mpi' | head
echo "== env"
env | grep -iE 'cuda|path' 
echo "== gcc"
gcc --version | head -1; g++ --version | head -1
echo "== nvidia-smi processes"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi -q | grep -iE 'compute mode|mig' | head
echo "== python"
python3 -m pip --version 2>&1 | head -1
ls /usr/bin/python3*
echo "== tmux"
which tmux screen
