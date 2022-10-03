#!/bin/bash
set -ex

# export environment
source build-env.sh

# copy source files to ani_csrc
cp external/torchani_sandbox/torchani/csrc/* ani_csrc/

# TODO we need to move this after lammps kokkos build
# build lammps-ani
rm -rf build; mkdir -p build; cd build
# cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_HEADER_DIR=${LAMMPS_ROOT}/src -DCUDNN_INCLUDE_PATH=${CONDA_PREFIX}/include -DCUDNN_LIBRARY_PATH=${CONDA_PREFIX}/lib ..
# For NGC PyTorch: we need to use the built-in PyTorch shared libraries, and we don't need to use custom cudnn
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_HEADER_DIR=${LAMMPS_ROOT}/src -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ${CUDNN_FLAGS} ..
make -j
cd ../
