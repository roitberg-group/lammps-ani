# LAMMPS-ANI

LAMMPS-ANI interface for large scale molecular dynamics simulation with ANI neural network potential.

## Quick Start

### Docker
```bash
docker pull ghcr.io/roitberg-group/lammps-ani:latest
docker run --gpus all -it ghcr.io/roitberg-group/lammps-ani:latest
cd /lammps-ani/examples/water && ./run.sh
```

### Singularity (HPC)
```bash
module load singularity
singularity pull -F docker://ghcr.io/roitberg-group/lammps-ani:latest
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES singularity exec --cleanenv -H ./:/home --nv lammps-ani_latest.sif /bin/bash
cp -r /lammps-ani ./lammps-ani
cd lammps-ani/examples/water && ./run.sh
```

**Note**: The pre-built container only supports Kokkos for Ada GPUs (L4, RTX 4090). For other GPUs, see [Container Guide](docs/container.md#re-build-within-container) to re-build within the container. For multi-GPU, recommend to build from source.


## Examples

- [water](examples/water/) - Simple water simulation
- [alanine-dipeptide](examples/alanine-dipeptide/) - Alanine dipeptide simulation
- [benchmark](examples/benchmark/) - Performance benchmark with water box
- [combustion](examples/combustion/) - Combustion reaction
- [early_earth](examples/early_earth/) - Early Earth chemistry simulation

Set environment before running: `source ./build-env.sh`

## Installation

### Requirements
- CUDA >= 12.8.1
- GCC >= 14.2.0
- PyTorch >= 2.8.0
- OpenMPI >= 5.0.7
- CMake >= 3.21.3

### Conda Environment
```bash
export conda_env_path=/path/to/env
conda create --prefix $conda_env_path python=3.11
conda activate $conda_env_path
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge libopenblas gsl -y
```

### Build from source

Following example is for HiPerGator L4 GPU node, adjust as needed for other systems.
```bash
srun --partition=hpg-turin --cpus-per-task=30 --gres=gpu:2 --mem=200gb -t 2:00:00 --pty /bin/bash -i
export conda_env_path=/blue/roitberg/apps/torch28/
conda activate $conda_env_path
module load cuda/12.8.1 gcc/14.2.0 openmpi/5.0.7 cmake/3.21.3

git clone --recursive git@github.com:roitberg-group/lammps-ani.git
# build
cd lammps-ani && ./build.sh
# export LAMMPS_ANI_ROOT environment variable; need to run this in every new shell
source ./build-env.sh
```

For additional build notes, see [Build Instructions](docs/build.md).

### Test
```bash
cd examples/water/
# Run a water box simulation
bash run.sh
```

## Usage

```lammps
pair_style     ani 5.1 model_file device [num_models] [ani_aev] [ani_neighbor] [ani_precision]
pair_coeff     * *
```

- `model_file` - Path to TorchANI model (.pt file)
- `device` - `cuda` or `cpu`
- `num_models` - Number of ensemble models, `-1` for all (default: `-1`)
- `ani_aev` - `cuaev` or `pyaev` (default: `cuaev`)
- `ani_neighbor` - `full` or `half` (default: `full`)
- `ani_precision` - `single` or `double` (default: `single`)

### Models

Available in `/lammps-ani/models/` (container) or export with `pytest models/test_models.py -s -v`:
- `ani2x.pt` - Standard ANI-2x
- `ani2x_repulsion.pt` - ANI-2x with repulsion
- `ani2x_ext0_repulsion.pt` - Extended ANI-2x with repulsion

For more usage details, see [Usage Instructions](docs/usage.md).

## Troubleshooting

**B200 GPUs hang**: `export UCX_NET_DEVICES=mlx5_0:1`

**Multi-GPU in containers**: Not recommended due to GPU Direct RDMA limitations. Build from source instead. See [#70](https://github.com/roitberg-group/lammps-ani/issues/70).
