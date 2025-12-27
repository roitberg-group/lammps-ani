# Container Guide

## Docker

```bash
docker pull ghcr.io/roitberg-group/lammps-ani:latest
docker run --gpus all -it ghcr.io/roitberg-group/lammps-ani:latest
cd /lammps-ani/examples/water && ./run.sh
```

### Re-build Within Docker

For non-Ada GPUs, re-build inside the container:

```bash
docker run --gpus all -it ghcr.io/roitberg-group/lammps-ani:latest
cd /lammps-ani && ./build.sh
```


## Singularity (HPC)

```bash
module load singularity
singularity pull -F docker://ghcr.io/roitberg-group/lammps-ani:latest
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES singularity exec --cleanenv -H ./:/home --nv lammps-ani_master.sif /bin/bash
cp -r /lammps-ani ./lammps-ani
cd lammps-ani/examples/water && ./run.sh
```

### Command Explanation

```bash
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES singularity exec --cleanenv -H ./:/home --nv lammps-ani_master.sif /bin/bash
```

- `SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES` - Pass GPU visibility into container
- `--cleanenv` - Don't inherit host environment variables
- `-H ./:/home` - Mount current directory as `/home` in container (for writable output)
- `--nv` - Enable NVIDIA GPU support

**Note**: The container's `/lammps-ani` is read-only. Use `-H` to mount a writable directory for simulation output.

### Re-Build Within Singularity

Pre-built container supports Kokkos for Ada (L4, RTX 4090) GPUs. For other GPUs, re-build inside the container:

```bash
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES singularity exec --cleanenv -H ./:/home --nv lammps-ani_master.sif /bin/bash

cp -r /lammps-ani ./lammps-ani
cd lammps-ani
export INSTALL_DIR=${HOME}/.local
./build.sh

export LAMMPS_ANI_ROOT=./
export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/
```

## Slurm Job with Singularity

```bash
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES singularity exec --cleanenv -H /path/to/workdir:/home --nv /path/to/lammps-ani_master.sif ./run.sh
```

## Limitations

Multi-GPU in containers is not recommended due to GPU Direct RDMA limitations. Build from source for multi-GPU. See [#70](https://github.com/roitberg-group/lammps-ani/issues/70).
