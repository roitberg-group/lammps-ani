#!/bin/bash
#SBATCH --job-name=lammps_ani        # Job name
#SBATCH --ntasks=1024                # Number of MPI tasks (i.e. processes)
#SBATCH --nodes=128                  # Maximum number of nodes to be allocated
#SBATCH --ntasks-per-node=8          # Maximum number of tasks on each node
#SBATCH --cpus-per-task=1            # Number of cores per MPI task
#SBATCH --partition=hpg-ai
#SBATCH --qos=roitberg
#SBATCH --account=roitberg
#SBATCH --reservation=roitberg-phase2
#SBATCH --gres=gpu:8
#SBATCH --mem=120gb                  # Memory (i.e. RAM) per processor
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jinzexue@ufl.edu # Where to send mail
#SBATCH --exclude=c0900a-s23
#SBATCH --time=150:00:00             # Wall time limit (days-hrs:min:sec)
#SBATCH --output=lammps_ani_22M_1024gpu_%j.log   # Path to the standard output and error files relative to the working dir

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load cuda/11.4.3 gcc/9.3.0 openmpi/4.1.5 cmake/3.21.3 git/2.30.1 
export LAMMPS_ANI_ROOT="/home/jinzexue/program/lammps-ani"
export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/jinzexue/program/torch1121 
echo using python: $(which python)

# please check ani_num_models! !!!!
python run_one.py data/mixture_22800000.data --kokkos --num_gpus=1024 --input_file=in.22M.lammps --log_dir=/red/roitberg/22M_20231222_prodrun --ani_model_file='ani1x_nr.pt' --run_name=early_earth_22M --ani_num_models=8 --timestep=0.25 --run

