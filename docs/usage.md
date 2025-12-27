# Usage

## TF32 Tensor Cores

Tensor cores (TF32 precision) are disabled by default to maintain force precision. To enable for faster computation:

```bash
export LAMMPS_ANI_ALLOW_TF32=1
```

## Slurm Job

```bash
#!/bin/bash
#SBATCH --partition=hpg-ai
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb

conda activate /blue/roitberg/apps/torch28
module load cuda/12.8.1 gcc/14.2.0 openmpi/5.0.7
source /path/to/lammps-ani/build-env.sh
# export UCX_NET_DEVICES=mlx5_0:1  # for B200

mpirun -np 1 ${LAMMPS_ROOT}/lmp_mpi -in in.lammps
```
