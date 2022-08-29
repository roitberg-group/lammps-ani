#!/bin/bash
set -ex

# files and envs
cp external/torchani_sandbox/torchani/csrc/* ani_csrc/
export lammps_root=${PWD}/external/lammps/
export LAMMPS_PLUGIN_PATH=${PWD}/build/
export CMAKE_CUDA_ARCHITECTURES="7.5;8.0"

# build torchani
cd external/torchani_sandbox
python setup.py develop --ext --user
pip install h5py ase
cd ../../

# save model
cd tests/test_ani2x_nocuaev_double/
python save_ani_nocuaev_double.py
cd ../../

# build lammps
cd external/lammps/
mkdir -p build-test; cd build-test
# D_GLIBCXX_USE_CXX11_ABI: https://stackoverflow.com/a/50873329/9581569
cmake -DCMAKE_C_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'  -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' -DLAMMPS_INSTALL_RPATH=yes -DPKG_GPU=no \
-DGPU_API=cuda -DGPU_ARCH=sm_80 -DPKG_PLUGIN=yes -DCMAKE_INSTALL_PREFIX=${HOME}/.local -DBUILD_MPI=yes -DBUILD_SHARED_LIBS=yes -DLAMMPS_MACHINE=mpi \
-DPKG_EXTRA-PAIR=on -DPKG_MOLECULE=on -DPKG_OPENMP=on -DENABLE_TESTING=on -DLAMMPS_EXCEPTIONS=on \
../cmake/
make -j
# test
mpirun -np 1 ${lammps_root}/build-test/test_pair_style ../unittest/force-styles/tests/mol-pair-lj_smooth.yaml
cd ../../../

# build lammps-ani
mkdir -p build; cd build
# cmake -DCMAKE_C_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' -DLAMMPS_HEADER_DIR=${lammps_root}/src -DCUDNN_INCLUDE_PATH=${CONDA_PREFIX}/include -DCUDNN_LIBRARY_PATH=${CONDA_PREFIX}/lib ..
cmake -DCMAKE_C_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' -DLAMMPS_HEADER_DIR=${lammps_root}/src ..
make -j
cd ../

# test
cd tests/
./test_all.sh
