#!/bin/bash
set -ex
NUM_TASKS=${SLURM_NTASKS:-1}
NUM_GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ $NUM_GPUs -gt 0 ]; then
    mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -in in.lammps.cuda
fi

# Domain Decomposition with 2 processes
if [ $NUM_TASKS -gt 1 ]; then
    if [ $NUM_GPUs -gt 0 ]; then
        mpirun -np 2 ${LAMMPS_ROOT}/build/lmp_mpi -in in.lammps.cuda
    fi
fi

rm log.lammps
rm water.final
