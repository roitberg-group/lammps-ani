#!/bin/bash
set -ex

# environment
# Paths
export LAMMPS_ANI_ROOT=${LAMMPS_ANI_ROOT:=${PWD}}  # default as PWD
export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/
# CUDA_ARCH
export CMAKE_CUDA_ARCHITECTURES="6.0+PTX;7.5;8.0"
export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.5;8.0"
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

# copy source files to ani_csrc
cp external/torchani_sandbox/torchani/csrc/* ani_csrc/

# build lammps-ani
rm -rf build; mkdir -p build; cd build
# cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_HEADER_DIR=${LAMMPS_ROOT}/src -DCUDNN_INCLUDE_PATH=${CONDA_PREFIX}/include -DCUDNN_LIBRARY_PATH=${CONDA_PREFIX}/lib ..
# For NGC PyTorch: we need to use the built-in PyTorch shared libraries, and we don't need to use custom cudnn
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_HEADER_DIR=${LAMMPS_ROOT}/src -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ${CUDNN_FLAGS} ..
make -j
cd ../
