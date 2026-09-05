# H100 environment for SpacePDHCG (CUDA 12.8 toolkit at /usr/local/cuda-12.8, driver already installed)
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH="$HOME/.local/bin:$HOME/.local/node/bin:/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export SPACEPDHCG_CUDA_ARCHITECTURES=90
export CUDA_VISIBLE_DEVICES=0
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.7rc1
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
