#!/bin/bash
#SBATCH --job-name=umbrella
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80gb
#SBATCH -t 120:00:00
#SBATCH --qos=roitberg
#SBATCH --account=roitberg
#SBATCH -o %j.out

module load cuda/11.4.3 gcc/9.3.0 openmpi/4.0.5 cmake/3.21.3 git/2.30.1 netcdf/4.7.2

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/roitberg/jinzexue/conda-envs/openmm 
echo $(which python)
packmol < mixture_228.inp 
