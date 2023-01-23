#!/bin/bash
set -ex

# export environment
source build-env.sh

# build torchani
cd external/torchani_sandbox
rm -rf build && python setup.py install --ext
pip install h5py ase pytest pyyaml
cd ../../

# save model
cd tests/
pytest save_ani.py -s -v
cd ../

# build kokkos
cd external/lammps
rm -rf build-kokkos; mkdir -p build-kokkos; cd build-kokkos
# kokkos does not support compiling for multiple GPU archs
# -DKokkos_ARCH_TURING75=yes -DKokkos_ARCH_PASCAL60=yes
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/kokkos -DBUILD_SHARED_LIBS=yes -DKokkos_ENABLE_CUDA=yes -DKokkos_ENABLE_OPENMP=yes -DKokkos_ENABLE_SERIAL=yes \
-DKokkos_ARCH_HOSTARCH=yes -DKokkos_ARCH_GPUARCH=on \
-DKokkos_ARCH_AMPERE80=yes -DKokkos_ENABLE_CUDA_LAMBDA=yes \
../lib/kokkos
make -j
make install
cd ..

# build lammps with unittest
rm -rf build-test; mkdir -p build-test; cd build-test
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_INSTALL_RPATH=yes \
-DCMAKE_PREFIX_PATH=${INSTALL_DIR}/kokkos \
-DPKG_PLUGIN=yes -DPKG_EXTRA-DUMP=yes -DBUILD_MPI=yes -DBUILD_SHARED_LIBS=yes -DLAMMPS_MACHINE=mpi \
-DPKG_EXTRA-PAIR=yes -DPKG_MOLECULE=yes -DPKG_OPENMP=yes -DENABLE_TESTING=yes -DLAMMPS_EXCEPTIONS=yes \
../cmake/
make -j
# test
mpirun -np 1 ${LAMMPS_ROOT}/build-test/test_pair_style ../unittest/force-styles/tests/mol-pair-morse.yaml
cd ..

# build lammps
# kokkos flag
KOKKOS_FLAGS="-DPKG_KOKKOS=yes -DEXTERNAL_KOKKOS=yes"
rm -rf build; mkdir -p build; cd build
# D_GLIBCXX_USE_CXX11_ABI: https://stackoverflow.com/a/50873329/9581569
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_INSTALL_RPATH=yes \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/lammps -DCMAKE_PREFIX_PATH=${INSTALL_DIR}/kokkos \
-DPKG_PLUGIN=yes -DPKG_EXTRA-DUMP=yes -DBUILD_MPI=yes -DBUILD_SHARED_LIBS=yes -DLAMMPS_MACHINE=mpi \
-DPKG_EXTRA-PAIR=no -DPKG_MOLECULE=no -DPKG_OPENMP=yes -DENABLE_TESTING=no -DLAMMPS_EXCEPTIONS=no \
$KOKKOS_FLAGS \
../cmake/
make -j ${MAKE_J_THREADS}
make install
cd ../../../
