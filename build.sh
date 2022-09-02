#!/bin/bash
set -ex

# export environment and build lammps-ani
source build-lammps-ani.sh

# build torchani
cd external/torchani_sandbox
rm -rf build && python setup.py install --ext
pip install h5py ase
cd ../../

# save model
cd tests/
python save_ani.py
cd ../

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

# # test
# cd tests/
# ./test_all.sh
