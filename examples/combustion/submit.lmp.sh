#!/bin/bash
#SBATCH --job-name=lammps_ani        # Job name
#SBATCH --ntasks=8                   # Number of MPI tasks (i.e. processes)
#SBATCH --nodes=1                    # Maximum number of nodes to be allocated
#SBATCH --ntasks-per-node=8          # Maximum number of tasks on each node
#SBATCH --cpus-per-task=1            # Number of cores per MPI task
#SBATCH --partition=hpg-ai
#SBATCH --qos=roitberg
#SBATCH --account=roitberg
#SBATCH --gres=gpu:8
#SBATCH --mem=100gb           # Memory (i.e. RAM) per processor
#SBATCH --time=150:00:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=lammps_ani_%j_1.log   # Path to the standard output and error files relative to the working dir

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load cuda/11.4.3 gcc/9.3.0 openmpi/4.1.5 cmake/3.21.3 git/2.30.1 
export LAMMPS_ANI_ROOT="/blue/roitberg/apps/lammps-ani"
export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/roitberg/apps/torch1121
echo using python: $(which python)

python run_one.py prepare_system/combustion-0.25-300k.data --kokkos --num_gpus=8 --input_file=in.lammps --log_dir=logs --ani_model_file='ani2x_repulsion.pt' --run_name=combustion_300k --ani_num_models=1 --timestep=0.5 --run_steps=2000000 --run
