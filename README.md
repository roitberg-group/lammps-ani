# LAMMPS-ANI
A plugin to run torchani on LAMMPS.  
On hipergator, the compiled program and a working example script could be found at `/blue/roitberg/apps/lammps-ani/examples/water/submit.sh`

## Requirement
Run an interactive session
```
srun --qos=roitberg --account=roitberg --nodes=1 --ntasks=2 --cpus-per-task=2 --mem=20gb --gres=gpu:2 --partition=hpg-ai -t 10:00:00 --pty /bin/bash -i
module load cuda/11.4.3 gcc/9.3.0 openmpi/4.0.5 cmake/3.21.3 git/2.30.1
export CMAKE_CUDA_ARCHITECTURES="7.5;8.0"
```

Build PyTorch from master branch: https://github.com/pytorch/pytorch#from-source

You could skip this step by using `conda activate /blue/roitberg/apps/cuda114/`
```bash
cd /some/path
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive --jobs 0

conda create -n lmp python=3.8
conda activate lmp
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda111

export TORCH_CUDA_ARCH_LIST="7.5;8.0"
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
```


Build LAMMPS
```bash
cd /some/path
git clone git@github.com:lammps/lammps.git
cd lammps
export lammps_root=${PWD}
mkdir build; cd build
cmake -DLAMMPS_INSTALL_RPATH=yes -DPKG_GPU=yes -DGPU_API=cuda -DGPU_ARCH=sm_80 -DPKG_PLUGIN=yes -DCMAKE_INSTALL_PREFIX=${HOME}/.local -DBUILD_MPI=yes -DBUILD_SHARED_LIBS=yes -DLAMMPS_MACHINE=mpi ../cmake/
make -j

# Optionally build with test
cd ..
mkdir build-test; cd build-test
cmake -DPKG_EXTRA-PAIR=on -DPKG_MOLECULE=on -DPKG_OPENMP=on -DENABLE_TESTING=on -DLAMMPS_EXCEPTIONS=on -DLAMMPS_INSTALL_RPATH=yes -DPKG_GPU=no -DGPU_API=cuda -DGPU_ARCH=sm_80 -DPKG_PLUGIN=yes -DCMAKE_INSTALL_PREFIX=${HOME}/.local -DBUILD_MPI=yes -DBUILD_SHARED_LIBS=yes ../cmake/
# run test
mpirun -np 1 ctest -V -R lj_smooth
# could also use the following to test
mpirun -np 1 ${lammps_root}/build-test/test_pair_style /path/to/mol-pair-lj_smooth.yaml
```

## Build lammps-ani
```bash
# build torchani
cd /some/path
git clone git@github.com:roitberg-group/torchani_sandbox.git
cd torchani_sandbox
git checkout withnbrlist
# skip the following line if you are using `conda activate /blue/roitberg/apps/cuda114/`
python setup.py develop --ext
cd ..

# lammps-ani
git clone git@github.com:roitberg-group/lammps-ani.git
cp torchani_sandbox/torchani/csrc/* lammps-ani/ani_csrc/
cd lammps-ani
mkdir build; cd build
cmake -DLAMMPS_HEADER_DIR=${lammps_root}/src -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"  ..
make -j
export LAMMPS_PLUGIN_PATH=${PWD}

# export a torchscript model
cd ../models
pip install torchvision --no-deps  # in case for import error
pip install h5py                   # in case for import error
python save_ani.py                 # you will get an ani2x_cuda.pt
cd ..
```

## Run example
make sure `LAMMPS_PLUGIN_PATH` and `lammps_root` are set correctly
```
export LAMMPS_PLUGIN_PATH=/blue/roitberg/apps/lammps-ani/build/
cd examples/water/
mpirun -np 8 ${lammps_root}/build/lmp_mpi -in in.lammps
```
