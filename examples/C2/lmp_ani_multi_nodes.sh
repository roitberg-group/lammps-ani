#!/bin/bash
#SBATCH --job-name=mpi_job_test      # Job name
#SBATCH --ntasks=80                   # Number of MPI tasks (i.e. processes)
#SBATCH --nodes=10                    # Maximum number of nodes to be allocated
#SBATCH --ntasks-per-node=8          # Maximum number of tasks on each node
#SBATCH --cpus-per-task=1            # Number of cores per MPI task 
#SBATCH --partition=hpg-ai
#SBATCH --reservation=hackathon
#SBATCH --account=nvidia-ai
#SBATCH --qos=nvidia-ai
#SBATCH --gpus=80
#SBATCH --mem-per-cpu=60gb          # Memory (i.e. RAM) per processor
#SBATCH --time=00:05:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=mpi_test_%j.log     # Path to the standard output and error files relative to the working directory

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load cuda/11.4.3 gcc/9.3.0 openmpi/4.0.5 cmake
cd /red/nvidia-ai/jinzexue/sp2022-hackathon/lammps-ani/example/C2
export LAMMPS_PLUGIN_PATH=/home/jinzexue/hackathon/sp2022-hackathon/lammps-ani/build
srun --mpi=pmix_v3 /red/nvidia-ai/jinzexue/lammps/build/lmp_mpi -in in.plugin.lammps 
