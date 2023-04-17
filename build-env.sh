#!/bin/bash

# environment
# Paths
export LAMMPS_ANI_ROOT=${LAMMPS_ANI_ROOT:=${PWD}}  # default as PWD
export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/
# Install Dir
export INSTALL_DIR=${INSTALL_DIR:=${HOME}/.local}  # default as $HOME/.local
# CUDA_ARCH
export CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES:="6.0+PTX;7.0;7.5;8.0"}
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
