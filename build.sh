#!/bin/bash
set -ex

# export environment
source build-env.sh

# build torchani
cd external/torchani_sandbox
rm -rf build && python setup.py install --ext
pip install h5py ase
cd ../../

# save model
cd tests/
python save_ani.py
cd ../

# build kokkos
# cd external/lammps/lib/kokkos
# rm -rf build; mkdir -p build; cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=~/.local/kokkos -DBUILD_SHARED_LIBS=yes -DKokkos_ENABLE_CUDA=yes -DKokkos_ENABLE_OPENMP=yes -DKokkos_ARCH_AMPERE80=yes # -DKokkos_ENABLE_SERIAL=yes -DKokkos_ARCH_TURING75=yes
# make -j
# make install
# cd ../../../../../

# build lammps
cd external/lammps/
# a patch that allows us to build kokkos as a shared library
git apply ../lammps-kokkos.patch
# kokkos flag
KOKKOS_FLAGS="-DPKG_KOKKOS=on -DEXTERNAL_KOKKOS=off -DKokkos_ARCH_HOSTARCH=yes -DKokkos_ARCH_GPUARCH=on -DKokkos_ENABLE_CUDA=yes -DKokkos_ENABLE_OPENMP=yes -DKokkos_ENABLE_SERIAL=no"
rm -rf build; mkdir -p build; cd build
# D_GLIBCXX_USE_CXX11_ABI: https://stackoverflow.com/a/50873329/9581569
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_INSTALL_RPATH=yes -DPKG_GPU=no \
-DGPU_API=cuda -DGPU_ARCH=sm_80 -DPKG_PLUGIN=yes -DCMAKE_INSTALL_PREFIX=${HOME}/.local/lammps -DBUILD_MPI=yes -DBUILD_SHARED_LIBS=yes -DLAMMPS_MACHINE=mpi \
-DPKG_EXTRA-PAIR=on -DPKG_MOLECULE=on -DPKG_OPENMP=on -DENABLE_TESTING=off -DLAMMPS_EXCEPTIONS=on \
$KOKKOS_FLAGS \
../cmake/
make -j
make install
cd ../
git apply --reverse ../lammps-kokkos.patch
# test
mpirun -np 1 ${LAMMPS_ROOT}/build/test_pair_style ../unittest/force-styles/tests/mol-pair-morse.yaml
cd ../../

# build lammps-ani
source build-lammps-ani.sh

# # test
# cd tests/
# ./test_all.sh
