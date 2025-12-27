#!/bin/bash
set -ex

# export environment
source build-env.sh

# copy source files to ani_csrc (only files, not directories)
cp external/torchani_sandbox/torchani/csrc/*.{cu,h,cpp,cuh} src/ani_csrc/

# build lammps-ani
# remove old building files
if [ -f "build/install_manifest.txt" ]; then
    echo "Found install_manifest.txt. Removing installed files..."
    xargs rm -vf < build/install_manifest.txt
fi
rm -rf build; mkdir -p build; cd build
# cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_HEADER_DIR=${LAMMPS_ROOT}/src -DCUDNN_INCLUDE_PATH=${CONDA_PREFIX}/include -DCUDNN_LIBRARY_PATH=${CONDA_PREFIX}/lib ..
# For NGC PyTorch: we need to use the built-in PyTorch shared libraries, and we don't need to use custom cudnn
# For debugging symbols, please add: -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" \
-DPython_EXECUTABLE=$(which python) \
-DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_STANDARD=17 \
-DLAMMPS_HEADER_DIR=${LAMMPS_ROOT}/src -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)');${INSTALL_DIR}/" \
-DCMAKE_INSTALL_LIBDIR=lib \
${CUDNN_FLAGS} ..
make -j
make install
cd ../
