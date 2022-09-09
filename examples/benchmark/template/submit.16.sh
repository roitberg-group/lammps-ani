#!/bin/bash
#SBATCH --job-name=lammps_ani        # Job name
#SBATCH --ntasks=16                   # Number of MPI tasks (i.e. processes)
#SBATCH --nodes=2                    # Maximum number of nodes to be allocated
#SBATCH --ntasks-per-node=8          # Maximum number of tasks on each node
#SBATCH --cpus-per-task=1            # Number of cores per MPI task
#SBATCH --partition=hpg-ai
#SBATCH --qos=roitberg
#SBATCH --account=roitberg
#SBATCH --gres=gpu:8
#SBATCH --mem-per-cpu=30gb           # Memory (i.e. RAM) per processor
#SBATCH --time=01:00:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=lammps_ani_%j.log   # Path to the standard output and error files relative to the working dir

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load cuda/11.4.3 gcc/9.3.0 openmpi/4.0.5 cmake
export LAMMPS_ANI_ROOT="/blue/roitberg/apps/lammps-ani"
export LAMMPS_PLUGIN_PATH="/blue/roitberg/apps/lammps-ani/build/"
export LAMMPS_ROOT="/blue/roitberg/apps/lammps-ani/external/lammps/"

srun --mpi=pmix_v3 ${LAMMPS_ROOT}/build/lmp_mpi -in in.lammps
