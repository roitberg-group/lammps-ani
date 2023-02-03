#!/bin/bash
set -ex
TIMESTAMP=`date +%F-%H%M`

# 20k or 300k
DIR=300k
NUM_MODELS=1
NUM_GPUS=8

mpirun -np ${NUM_GPUS} ${LAMMPS_ROOT}/build/lmp_mpi \
    -var newton_pair off -var num_models ${NUM_MODELS} -var datafile ${DIR}/water.data \
    -log ${DIR}/log-${TIMESTAMP}-models_${NUM_MODELS}-gpus_${NUM_GPUS}.log -in in.lammps
