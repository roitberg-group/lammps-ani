#!/bin/bash
#SBATCH --job-name=tip3p_shake
#SBATCH --output=run1/slurm_%j.out
#SBATCH --error=run1/slurm_%j.err
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
#SBATCH --qos=roitberg

# TIP3P Water Run 1: SHAKE constraints
# 15 ns total (5 ns NPT + 10 ns NVT) at 1.0 fs timestep
# 15,000,000 steps

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

module load cuda/12.8.1 gcc/14.2.0 openmpi/5.0.7 lammps/29Aug24

# cd $SLURM_SUBMIT_DIR

lmp_kokkos_cuda -sf gpu -pk gpu 1 -in in.run1.lammps -log run1/output.log

echo "Job finished at $(date)"
