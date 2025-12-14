#!/bin/bash
#SBATCH --job-name=tip3p_flex
#SBATCH --output=run2/slurm_%j.out
#SBATCH --error=run2/slurm_%j.err
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=50gb
#SBATCH --time=24:00:00
#SBATCH --qos=roitberg

# TIP3P Water Run 2: Flexible (no SHAKE)
# 15 ns total (5 ns NPT + 10 ns NVT) at 0.2 fs timestep
# 75,000,000 steps (~5x longer than Run 1)

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

module load cuda/12.8.1 gcc/14.2.0 openmpi/5.0.7 lammps/29Aug24

# cd $SLURM_SUBMIT_DIR

lmp_kokkos_cuda -sf gpu -pk gpu 1 -in in.run2.lammps -log run2/output.log

echo "Job finished at $(date)"
