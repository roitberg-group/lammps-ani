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
export CXX11_ABI=1

# copy source files to ani_csrc
cp external/torchani_sandbox/torchani/csrc/* ani_csrc/

# build torchani
cd external/torchani_sandbox
rm -rf build && python setup.py install --ext
pip install h5py ase
cd ../../

# save model
cd tests/test_ani2x_nocuaev_double/
python save_ani_nocuaev_double.py
cd ../../

# build lammps
cd external/lammps/
rm -rf build; mkdir -p build; cd build
# D_GLIBCXX_USE_CXX11_ABI: https://stackoverflow.com/a/50873329/9581569
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_INSTALL_RPATH=yes -DPKG_GPU=no \
-DGPU_API=cuda -DGPU_ARCH=sm_80 -DPKG_PLUGIN=yes -DCMAKE_INSTALL_PREFIX=${HOME}/.local -DBUILD_MPI=yes -DBUILD_SHARED_LIBS=yes -DLAMMPS_MACHINE=mpi \
-DPKG_EXTRA-PAIR=on -DPKG_MOLECULE=on -DPKG_OPENMP=on -DENABLE_TESTING=on -DLAMMPS_EXCEPTIONS=on \
../cmake/
make -j
# test
mpirun -np 1 ${LAMMPS_ROOT}/build/test_pair_style ../unittest/force-styles/tests/mol-pair-lj_smooth.yaml
cd ../../../

# build lammps-ani
rm -rf build; mkdir -p build; cd build
# cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_HEADER_DIR=${LAMMPS_ROOT}/src -DCUDNN_INCLUDE_PATH=${CONDA_PREFIX}/include -DCUDNN_LIBRARY_PATH=${CONDA_PREFIX}/lib ..
# For NGC PyTorch: we need to use the built-in PyTorch shared libraries, and we don't need to use custom cudnn
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_HEADER_DIR=${LAMMPS_ROOT}/src -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make -j
cd ../

# # test
# cd tests/
# ./test_all.sh
