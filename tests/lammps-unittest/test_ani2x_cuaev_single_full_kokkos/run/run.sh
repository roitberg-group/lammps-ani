#!/bin/bash
set -ex
NUM_TASKS=${SLURM_NTASKS:-1}
NUM_GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ $NUM_GPUs -gt 0 ]; then
    mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -var num_models 8 -in in.lammps.cuda
    LAMMPS_ANI_PROFILING=1 mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -k on g 1 -sf kk -var newton_pair on -var num_models 8 -in in.lammps.cuda
fi

# Domain Decomposition with 2 processes
if [ $NUM_TASKS -gt 1 ]; then
    if [ $NUM_GPUs -gt 0 ]; then
        mpirun -np 2 ${LAMMPS_ROOT}/build/lmp_mpi -var num_models 8 -in in.lammps.cuda
        LAMMPS_ANI_PROFILING=1 mpirun -np 2 ${LAMMPS_ROOT}/build/lmp_mpi -k on g 2 -sf kk -var newton_pair on -var num_models 8 -in in.lammps.cuda
    fi
fi

rm log.lammps
rm water.final
