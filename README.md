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

## Benchmark
Use the pre-built docker container
```bash
docker pull ghcr.io/roitberg-group/lammps-ani:master
docker run --gpus all -it ghcr.io/roitberg-group/lammps-ani:master
cd /lammps-ani/examples/water/
# Run benchmark script. Note that for this container, Kokkos is only available for A100 GPUs
bash benchmark.sh
```

## Docker container
You could use the pre-built [docker container](https://github.com/roitberg-group/lammps-ani/pkgs/container/lammps-ani) to avoid compiling the program by yourself. The pre-built container only supports Kokkos for A100 GPUs.
```bash
docker pull ghcr.io/roitberg-group/lammps-ani:master
docker run --gpus all -it ghcr.io/roitberg-group/lammps-ani:master
```

## Singularity container
Some HPCs provide Singularity instead of Docker. The following shows the instruction for Singularity usage:
```bash
git clone --recursive git@github.com:roitberg-group/lammps-ani.git
singularity pull -F docker://ghcr.io/roitberg-group/lammps-ani:master
mkdir -p ~/singularity-home
# exec into container
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES singularity exec --cleanenv -H ~/singularity-home:/home --nv lammps-ani_master.sif /bin/bash
# test
cd lammps-ani
nvidia-smi && cd external/torchani_sandbox && python setup.py install --ext --user && cd ../../ && cd tests/ && pytest test_models.py -s -v && ./test_all.sh
```

## Build from source
```bash
git clone --recursive git@github.com:roitberg-group/lammps-ani.git
./build.sh
# by default this will build CUAEV for all GPU architectures, you could speed up this process by specifying
# the CMAKE_CUDA_ARCHITECTURES for specific architectures, for example:
# CMAKE_CUDA_ARCHITECTURES="7.5;8.0" ./build.sh
```

Users need to set environment variables, please check [run examples](#run-examples) section.

## Run examples
There are 3 environment variables needed to run LAMMPS-ANI.
- For docker and singularity contianer, these variables are already set.
- For local build users, they need to set them correctly and put them into `~/.bashrc` file and also the script file before submit slurm jobs. You could also run `source ./build-env.sh` to load them automatically.
```bash
export LAMMPS_ANI_ROOT=<PATH_TO_LAMMPS_ANI_ROOT>
export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/
```

There are several simulation examples under [examples](examples/) folder.
