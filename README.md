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

pytorch and cudnn
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge cudnn=8.3.2
```


Build LAMMPS
```bash
cd /some/path
git clone git@github.com:lammps/lammps.git
cd lammps
export LAMMPS_ROOT=${PWD}
mkdir build; cd build
# D_GLIBCXX_USE_CXX11_ABI: https://stackoverflow.com/a/50873329/9581569
cmake -DCMAKE_C_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'  -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' -DLAMMPS_INSTALL_RPATH=yes -DPKG_GPU=no \
-DGPU_API=cuda -DGPU_ARCH=sm_80 -DPKG_PLUGIN=yes -DCMAKE_INSTALL_PREFIX=${HOME}/.local -DBUILD_MPI=yes -DBUILD_SHARED_LIBS=yes -DLAMMPS_MACHINE=mpi \
-DPKG_EXTRA-PAIR=on -DPKG_MOLECULE=on -DPKG_OPENMP=on -DENABLE_TESTING=on -DLAMMPS_EXCEPTIONS=on \
../cmake/
make -j
# run test
ctest -V -R lj_smooth
# could also use the following to test
${LAMMPS_ROOT}/build-test/test_pair_style ../unittest/force-styles/tests/mol-pair-lj_smooth.yaml
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
git clone --recursive git@github.com:roitberg-group/lammps-ani.git
cp torchani_sandbox/torchani/csrc/* lammps-ani/ani_csrc/
cd lammps-ani
mkdir build; cd build
cmake -DCMAKE_C_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' -DLAMMPS_HEADER_DIR=${LAMMPS_ROOT}/src -DCUDNN_INCLUDE_PATH=${CONDA_PREFIX}/include -DCUDNN_LIBRARY_PATH=${CONDA_PREFIX}/lib ..
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
make sure `LAMMPS_PLUGIN_PATH` and `LAMMPS_ROOT` are set correctly
```
export LAMMPS_PLUGIN_PATH=/blue/roitberg/apps/lammps-ani/build/
cd examples/water/
mpirun -np 8 ${LAMMPS_ROOT}/build/lmp_mpi -in in.lammps
```

## Singularity Usage
```bash
git clone --recursive git@github.com:roitberg-group/lammps-ani.git
singularity pull -F docker://ghcr.io/roitberg-group/lammps-ani:master
mkdir -p ~/singularity-home
# exec into container
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES singularity exec --cleanenv -H ~/singularity-home:/home --nv lammps-ani_master.sif /bin/bash
# test
cd lammps-ani
nvidia-smi && cd external/torchani_sandbox && python setup.py install --ext --user && cd ../../ && cd tests/test_ani2x_nocuaev_double && python save_ani_nocuaev_double.py && cd ../ && ./test_all.sh
```
