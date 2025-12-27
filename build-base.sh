#!/bin/bash
set -ex

# export environment
source build-env.sh

# Check if the python installation directory is writable
dir_to_check=$(python -c 'import site; print(site.getsitepackages()[0])')
temp_file="$dir_to_check/.write_test_$(date +%s%N)"
# Temporarily disable 'set -e'
set +e
# Attempt to create the temporary file
touch "$temp_file" > /dev/null 2>&1
# Check the result of the touch command to determine writability
if [ $? -eq 0 ]; then
    # If writable, set the variable to an empty string
    install_option=""
    # Clean up the temporary file
    rm -f "$temp_file"
else
    # If not writable, set the variable to '--user'
    install_option="--user"
fi
# Re-enable 'set -e'
set -e

# build torchani
cd external/torchani_sandbox
# rm -rf build && pip install -e . --config-settings=--global-option=ext --no-build-isolation --no-deps -v $install_option
# Force extension build through environment variable instead of command line arg
export TORCHANI_BUILD_EXTENSIONS=1
# export TORCHANI_BUILD_ALL_EXTENSIONS=1
rm -rf build && pip install -e . --no-build-isolation -v $install_option
pip install h5py ase pytest pyyaml --upgrade $install_option
cd ../../

# install ani_ext
cd external/ani_ext
python setup.py develop $install_option
cd ../../

# install ani_engine
cd external/ani_engine
pip install -e . $install_option 
cd ../../

# save model
cd tests/
pytest test_models.py -s -v
cd ../

# build kokkos
cd external/lammps
# remove old building files
if [ -f "build-kokkos/install_manifest.txt" ]; then
    echo "Found install_manifest.txt. Removing installed files..."
    xargs rm -vf < build-kokkos/install_manifest.txt
fi
rm -rf build-kokkos; mkdir -p build-kokkos; cd build-kokkos
# get current gpu architecture by using pytorch to get device capability
KOKKOS_ARCH=${OVERRIDE_KOKKOS_ARCH:=$(python -c "import torch; gpu_sm = ''.join(map(str, torch.cuda.get_device_capability(0))); gpu_name = torch.cuda.get_device_name(0); kokkos_dict = {'70': 'Kokkos_ARCH_VOLTA70', '75': 'Kokkos_ARCH_TURING75', '80': 'Kokkos_ARCH_AMPERE80', '86': 'Kokkos_ARCH_AMPERE86', '89': 'Kokkos_ARCH_ADA89', '90': 'Kokkos_ARCH_HOPPER90'}; kokkos_arch = kokkos_dict[gpu_sm]; print(kokkos_arch)")}
echo Building KOKKOS for ${KOKKOS_ARCH}
# kokkos does not support compiling for multiple GPU archs
# -DKokkos_ARCH_TURING75=yes -DKokkos_ARCH_PASCAL60=yes
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/ -DCMAKE_INSTALL_LIBDIR=lib \
-DBUILD_SHARED_LIBS=yes -DKokkos_ENABLE_CUDA=yes -DKokkos_ENABLE_OPENMP=yes -DKokkos_ENABLE_SERIAL=yes \
-DKokkos_ARCH_HOSTARCH=yes -DKokkos_ARCH_GPUARCH=on \
-D${KOKKOS_ARCH}=yes -DKokkos_ENABLE_CUDA_LAMBDA=yes \
../lib/kokkos
make -j
make install
cd ..

# build lammps with unittest
rm -rf build-test; mkdir -p build-test; cd build-test
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DLAMMPS_INSTALL_RPATH=yes \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_PREFIX_PATH=${INSTALL_DIR}/ \
-DPKG_PLUGIN=yes -DPKG_EXTRA-DUMP=yes -DBUILD_MPI=yes -DBUILD_SHARED_LIBS=yes -DLAMMPS_MACHINE=mpi \
-DPKG_EXTRA-PAIR=yes -DPKG_MOLECULE=yes -DENABLE_TESTING=yes -DLAMMPS_EXCEPTIONS=yes \
-DPKG_OPENMP=yes \
../cmake/
make -j
# test
mpirun -np 1 ${LAMMPS_ROOT}/build-test/test_pair_style ../unittest/force-styles/tests/mol-pair-morse.yaml
cd ..

# build lammps
# kokkos flag
KOKKOS_FLAGS="-DPKG_KOKKOS=yes -DEXTERNAL_KOKKOS=yes"
# remove old building files
if [ -f "build/install_manifest.txt" ]; then
    echo "Found install_manifest.txt. Removing installed files..."
    xargs rm -vf < build/install_manifest.txt
fi
rm -rf build; mkdir -p build; cd build
# D_GLIBCXX_USE_CXX11_ABI: https://stackoverflow.com/a/50873329/9581569
# Ensure conda libs are in library path for PLUMED dependencies (gsl, mkl, zlib)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
cmake -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}" \
-DCMAKE_CXX_STANDARD=17 -DLAMMPS_INSTALL_RPATH=yes \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/ -DCMAKE_PREFIX_PATH="${INSTALL_DIR}/;${CONDA_PREFIX}/lib;${CONDA_PREFIX}/include" -DCMAKE_INSTALL_LIBDIR=lib \
-DPKG_PLUGIN=yes -DPKG_EXTRA-DUMP=yes -DBUILD_MPI=yes -DBUILD_SHARED_LIBS=yes -DLAMMPS_MACHINE=mpi \
-DPKG_EXTRA-PAIR=no -DPKG_MOLECULE=yes -DPKG_RIGID=yes -DPKG_KSPACE=yes -DPKG_COLVARS=yes -DPKG_PLUMED=no -DDOWNLOAD_PLUMED=yes \
-DENABLE_TESTING=no -DLAMMPS_EXCEPTIONS=yes \
-DPKG_OPENMP=yes \
$KOKKOS_FLAGS \
../cmake/
make -j ${MAKE_J_THREADS}
make install
cd ../../../
