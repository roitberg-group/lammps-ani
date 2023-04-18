# LAMMPS-ANI
A plugin to run torchani on LAMMPS.

## Requirement
### For Hipergator users
```bash
# run an interactive session
srun --qos=roitberg --account=roitberg --nodes=1 --ntasks=2 --cpus-per-task=2 --mem=80gb --gres=gpu:2 --partition=hpg-ai -t 3:00:00 --pty /bin/bash -i
# load modules
module load cuda/11.4.3 gcc/9.3.0 openmpi/4.0.5 cmake/3.21.3 git/2.30.1 netcdf/4.7.2 singularity
```

PyTorch and cuDNN
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c conda-forge cudnn=8.3.2
```

Netcdf should be avialable in most HPC systems, and could be loaded using `module load` command. Otherwise, if you are running Ubuntu, you could install it by
```
apt install libnetcdf-dev
```


### For Expanse users

Expanse does not have the latest CUDA libraries to compile Lammps-ANI, please use the [singularity](#singularity-container) container instead

```bash
srun -p gpu-shared --nodes=1 --ntasks=2 --account=[YOUR_ACCOUNT_NAME] --cpus-per-task=2 --gpus=2 --time=03:00:00 --mem=80gb --pty -u bash -i
module load singularitypro
```

Please check [Singularity container](#singularity-container) section on how to use it.

## Docker container
You could use the pre-built [docker container](https://github.com/roitberg-group/lammps-ani/pkgs/container/lammps-ani) to avoid compiling the program by yourself. The pre-built container only supports Kokkos for A100 GPUs.
```bash
docker pull ghcr.io/roitberg-group/lammps-ani:master
docker run --gpus all -it ghcr.io/roitberg-group/lammps-ani:master
```

## Singularity container
Some HPCs provide Singularity instead of Docker. The following shows the instruction for Singularity usage:

``` bash
# First load singularity module
module load singularity  # or `module load singularitypro` for Expanse
# Setup Cache and TMP directory for singularity, this step is needed for Expanse Users
export SINGULARITY_CACHEDIR=/scratch/$USER/job_$SLURM_JOB_ID
export SINGULARITY_TMPDIR=/scratch/$USER/job_$SLURM_JOB_ID
# Create and enter a new folder
mkdir ani; cd ani
# Pull the container
singularity pull -F docker://ghcr.io/roitberg-group/lammps-ani:master
# Clone the repo
git clone --recursive git@github.com:roitberg-group/lammps-ani.git
# The folder structure now should be like:
# .
# ├── lammps-ani
# └── lammps-ani_master.sif
# Exec into the container
# [TODO] slurm environment might also be needed
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES singularity exec --cleanenv -H ./:/home --nv lammps-ani_master.sif /bin/bash

# Then run the water example
cd lammps-ani/examples/water
# If you don't have A100 GPUs, you need to change the `RUN_KOKKOS` variable to `no` in the `run.sh` script.
# Run it
./run.sh
```

You could also build kokkos for the GPUs you have, which moves all the LAMMPS internal computations such as neighbor list calculation, position update, etc to GPUs. It also eliminates the need for CPU-GPU data transfer and significantly improves the performance. Please check section [build within container](#build-within-container) for more detail.

Please also check the [alanine dipeptide](examples/alanine-dipeptide) example.

## Usage
```bash
pair_style     ani 5.1 model_file device num_models ani_aev ani_neighbor
pair_coeff     * *
```
0. `model_file`               = path to the model file
1. `device`                   = `cuda`/`cpu`
2. `num_models` (Optional)    = number of models to use, default as `-1` to use all models within the ensemble
3. `ani_aev` (Optional)       = `cuaev`/`pyaev`, default as `cuaev`
4. `ani_neighbor` (Optional)  = `full`/`half`, default as full nbrlist. Note that full nbrlist is prefered for performance benefit.
5. `ani_precision` (Optional) = `single`/`double`, default as single


## Models

Currently there are 3 models available, `ani2x.pt`, `ani2x_repulsion.pt`, `ani2x_ext0_repulsion.pt`. They are defined at [models.py](tests/models.py) and tested by [test_models.py](tests/test_models.py). To use custom models, the models need to follow a strict format, user could check the models defined at the [models.py](tests/models.py) for more detail.


```bash
cd lammps-ani
# When using singularity container, you may encounter write permission error, you could solve it by install torchani_sandbox to your home directory by:
# cd external/torchani_sandbox && python setup.py install --ext --user && cd ../../
cd tests/
# save models and tests
pytest test_models.py -s -v

# a longer tests if you want to run
pytest test_lmp_with_ase.py -s -v
```

## Build from source
```bash
git clone --recursive git@github.com:roitberg-group/lammps-ani.git
./build.sh
# by default this will build CUAEV for all GPU architectures, you could speed up this process by specifying
# the CMAKE_CUDA_ARCHITECTURES for specific architectures, for example:
# CMAKE_CUDA_ARCHITECTURES="7.5;8.0" ./build.sh
```

User could use the absolute path to run lammps (please check [run examples](#run-examples) section), or install lammps binaries and libraries into `${HOME}/.local`. For the later case, you need to set `PATH` and `LD_LIBRARY_PATH` environment variables to be able to use `lmp_mpi` directly.
```bash
export PATH=${HOME}/.local/bin:$PATH
export LD_LIBRARY_PATH=${HOME}/.local/lib:$LD_LIBRARY_PATH
export LAMMPS_PLUGIN_PATH=${HOME}/.local/lib
```

test
```bash
lmp_mpi -help
# run without kokkos
mpirun -np 1 lmp_mpi ... -in in.lammps
# run with kokkos
mpirun -np 1 lmp_mpi -k on g 1 -sf kk -pk kokkos gpu/aware on ... -in in.lammps
```

## Build within container


## Run examples
There are 3 environment variables needed to run LAMMPS-ANI with absolute path.
- For docker and singularity contianer, these variables are already set.
- For local build users, they need to set them correctly and put them into `~/.bashrc` file and also the script file before submit slurm jobs. You could also run `source ./build-env.sh` to load them automatically.
```bash
export LAMMPS_ANI_ROOT=<PATH_TO_LAMMPS_ANI_ROOT>
export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/
```

There are several simulation examples under [examples](examples/) folder.

## Benchmark

Use the pre-built docker container
```bash
docker pull ghcr.io/roitberg-group/lammps-ani:master
docker run --gpus all -it ghcr.io/roitberg-group/lammps-ani:master
cd /lammps-ani/examples/water/
# Run benchmark script. Note that for this container, Kokkos is only available for A100 GPUs
bash benchmark.sh
```
