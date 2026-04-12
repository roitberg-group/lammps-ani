#!/bin/bash

# environment
# Paths
export LAMMPS_ANI_ROOT=${PWD}  # use current directory
export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/
# Install Dir
export INSTALL_DIR=${INSTALL_DIR:=${HOME}/.local}  # default as $HOME/.local
# CUDA_ARCH
export CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES:="6.0+PTX;7.0;7.5;8.0;9.0;10.0"}
export TORCH_CUDA_ARCH_LIST=${CMAKE_CUDA_ARCHITECTURES}
# NGC PyTorch needs CXX11_ABI
export CXX11_ABI=${CXX11_ABI:=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")}

# CUDNN_FLAGS
CUDNN_PATH=$(whereis cudnn | awk '{print $2}')
if [ -z "${CUDNN_PATH}" ]; then
    echo "CUDNN not found, will use the conda CUDNN"
    CUDNN_FLAGS="-DCUDNN_INCLUDE_PATH=${CONDA_PREFIX}/include -DCUDNN_LIBRARY_PATH=${CONDA_PREFIX}/lib"
else
    CUDNN_FLAGS=" "
fi

# Build Options
export MAKE_J_THREADS=${MAKE_J_THREADS:=""}  # default as all threads
export OVERRIDE_KOKKOS_ARCH=${OVERRIDE_KOKKOS_ARCH:=""}  # default as null

# Blackwell (SM 100/120) nodes require UCX_NET_DEVICES=mlx5_0:1 to avoid IB crash
# Auto-detect and set only on Blackwell GPUs; users can still override manually
_GPU_SM=$(python -c "import torch; print(''.join(map(str, torch.cuda.get_device_capability(0))))" 2>/dev/null || echo "")
if [[ "$_GPU_SM" == "100" || "$_GPU_SM" == "120" ]]; then
    export UCX_NET_DEVICES=${UCX_NET_DEVICES:=mlx5_0:1}
fi
unset _GPU_SM
