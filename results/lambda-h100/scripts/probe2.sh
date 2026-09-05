#!/bin/bash
echo "== /usr/lib/cuda layout"
ls /usr/lib/cuda
ls /usr/lib/cuda/lib64 2>/dev/null | head
echo "== which nvcc"; which nvcc; readlink -f "$(which nvcc)"
echo "== dpkg -S /usr/local/cuda"; dpkg -S /usr/local/cuda 2>&1 | head -3
ls -la /usr/local/cuda
echo "== cuda libs in system dirs"
ls /usr/lib/x86_64-linux-gnu | grep -E '^lib(cublas|cusparse|cusolver|cudart|curand|nvrtc|cufft)' | head -20
ls /usr/include | grep -E '^(cublas_v2|cusparse|cusolver|cuda_runtime|cuda)\.h' 
echo "== apt policy cuda repo"
apt-cache policy 2>/dev/null | grep -iE 'developer.download.nvidia|cuda' | head -5
apt-cache policy python3.12 2>/dev/null | head -3
apt-cache policy ninja-build jq 2>/dev/null | grep -E 'Candidate' 
echo "== disk"
df -h / /home /tmp | tail -3
echo "== compute-sanitizer version"; compute-sanitizer --version 2>&1 | head -2
echo "== dpkg -S compute-sanitizer"; dpkg -S /usr/bin/compute-sanitizer 2>&1 | head -1
echo "== nvidia-cuda-toolkit files sample"; dpkg -L nvidia-cuda-toolkit | grep -E 'bin/|nsys|nsight' | head -20
echo "== ld.so.conf"; cat /etc/ld.so.conf.d/*.conf 2>/dev/null | head
echo "== ssh keepalive"; grep -iE 'ClientAlive' /etc/ssh/sshd_config | head
echo "== internet"; curl -sI https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ | head -1
curl -sI https://github.com | head -1
