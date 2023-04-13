# LAMMPS-ANI
A plugin to run torchani on LAMMPS.

## Requirement
Run an interactive session on hipergator
```
srun --qos=roitberg --account=roitberg --nodes=1 --ntasks=2 --cpus-per-task=2 --mem=20gb --gres=gpu:2 --partition=hpg-ai -t 10:00:00 --pty /bin/bash -i
module load cuda/11.4.3 gcc/9.3.0 openmpi/4.0.5 cmake/3.21.3 git/2.30.1 netcdf/4.7.2 singularity
```

pytorch and cudnn
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c conda-forge cudnn=8.3.2
```

Netcdf should be avialable in most HPC systems, and could be loaded using `module load` command. Otherwise, if you are running Ubuntu, you could install it by
```
apt install libnetcdf-dev
```

# Benchmark
Pre-built docker container (kokkos only works for A100 GPUs).
```
docker pull ghcr.io/roitberg-group/lammps-ani-pre:0.0.1
docker run --gpus all -it ghcr.io/roitberg-group/lammps-ani-pre:0.0.1
cd /lammps-ani/examples/water/
# run benchmark script
# Note that for this container, Kokkos is only available for A100 GPUs
bash benchmark.sh
```

## Singularity & Docker Container
You could use the pre-built [docker container](https://github.com/roitberg-group/lammps-ani/pkgs/container/lammps-ani) to avoid compiling the program by yourself.

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

## Build
```bash
./build.sh
```

## Run example
make sure `LAMMPS_PLUGIN_PATH` and `LAMMPS_ROOT` are set correctly
```
export LAMMPS_PLUGIN_PATH=/blue/roitberg/apps/lammps-ani/build/
cd examples/water/
mpirun -np 8 ${LAMMPS_ROOT}/build/lmp_mpi -in in.lammps
```
