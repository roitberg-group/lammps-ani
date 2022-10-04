#!/bin/bash
set -ex

NUM_GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ $NUM_GPUs -gt 0 ]; then
    mpirun -np 1 ${LAMMPS_ROOT}/build-test/test_pair_style manybody-pair-ani-pbc-single-cuda.yaml -d ./ # -s
    mpirun -np 1 ${LAMMPS_ROOT}/build-test/test_pair_style manybody-pair-ani-single-cuda.yaml -d ./ # -s
fi
