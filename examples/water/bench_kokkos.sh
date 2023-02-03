#!/bin/bash
set -ex
TIMESTAMP=`date +%F-%H%M`

# 20k or 300k
DIR=20k
NUM_MODELS=1
NUM_GPUS=1

# LAMMPS_ANI_PROFILING=1 is only for profiling purpose to show the correct timing breakdown
LAMMPS_ANI_PROFILING=1 mpirun -np ${NUM_GPUS} ${LAMMPS_ROOT}/build/lmp_mpi \
    -k on g ${NUM_GPUS} -sf kk -pk kokkos gpu/aware on \
    -var newton_pair on -var num_models ${NUM_MODELS} -var datafile ${DIR}/water.data \
    -log ${DIR}/log-${TIMESTAMP}-kokkos-models_${NUM_MODELS}-gpus_${NUM_GPUS}.log -in in.lammps






