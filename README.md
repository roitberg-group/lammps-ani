# LAMMPS-ANI
A plugin to run torchani on LAMMPS.

## Requirement
Run an interactive session
```
srun --partition=hpg-ai --reservation=hackathon --account=nvidia-ai --qos=nvidia-ai --gpus=8 --nodes=1 --ntasks=8 --cpus-per-task=25 --mem=80gb -t 10:00:00 --pty /bin/bash -i
module load cuda/11.4.3 gcc/9.3.0 openmpi/4.0.5 cmake/3.21.3 git/2.30.1
export CMAKE_CUDA_ARCHITECTURES="7.5;8.0"
```

Build PyTorch from master branch: https://github.com/pytorch/pytorch#from-source

You could skip this step by using `conda activate /red/nvidia-ai/roitberg-group/lmp`
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
cmake -DLAMMPS_INSTALL_RPATH=yes -DPKG_PLUGIN=yes -DCMAKE_INSTALL_PREFIX=${HOME}/.local -DBUILD_MPI=yes -DBUILD_SHARED_LIBS=yes -DLAMMPS_MACHINE=mpi ../cmake/
make -j
```

## Build lammps-ani
```bash
cd /some/path

# lammps-ani
git clone git@github.com:roitberg-group/sp2022-hackathon.git
cd sp2022-hackathon/

# build torchani
git clone git@github.com:roitberg-group/torchani_sandbox.git
cd torchani_sandbox
git checkout withnbrlist
# skip the following line if you are using `conda activate /red/nvidia-ai/roitberg-group/lmp`
python setup.py develop --ext
cd ..

cp torchani_sandbox/torchani/csrc/* lammps-ani/ani_csrc/

cd lammps-ani
mkdir build; cd build
cmake -DLAMMPS_HEADER_DIR=${lammps_root}/src -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"  ..
make -j
export LAMMPS_PLUGIN_PATH=${PWD}

cd ..
pip install torchvision --no-deps  # in case for import error
pip install h5py                   # in case for import error
python save_ani.py                 # you will get an ani2x_cuda.pt
```


## Run example
make sure `LAMMPS_PLUGIN_PATH` and `lammps_root` are set correctly
```
export UCX_NET_DEVICES=mlx5_0:1
cp ani2x_cuda.pt example/water/
cd example/water/
mpirun -np 8 ${lammps_root}/build/lmp_mpi -in in.plugin.lammps
```
