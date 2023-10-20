#!/bin/sh

#SBATCH --account             cwr109
#SBATCH --cpus-per-task       1
#SBATCH --partition           gpu-shared
#SBATCH --gpus                1
#SBATCH --job-name            umbrella 
#SBATCH --mem-per-cpu         50gb
#SBATCH --nodes               1
#SBATCH --ntasks-per-node     1
#SBATCH --ntasks              1
#SBATCH --output              lammps_ani_%j_1GPUs.log
#SBATCH --partition           gpu-shared
#SBATCH --time                48:00:00
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Nodes Allocated                = $SLURM_JOB_NODELIST"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
module load gpu/0.15.4 openmpi/4.0.4 cuda11.7/toolkit/11.7.1 cmake/3.19.8 netcdf-c/4.7.4 singularitypro
export LAMMPS_ANI_ROOT="/home/richardx/dev/lammps-ani"
export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/

source $(conda info --base)/etc/profile.d/conda.sh
conda activate torch1121 
echo using python: $(which python)

