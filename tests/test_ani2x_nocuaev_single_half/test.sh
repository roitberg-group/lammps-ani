#!/bin/bash
set -ex

NUM_GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

mpirun -np 1 ${LAMMPS_ROOT}/build-test/test_pair_style manybody-pair-ani-single-cpu.yaml -d ./ # -s
mpirun -np 1 ${LAMMPS_ROOT}/build-test/test_pair_style manybody-pair-ani-pbc-single-cpu.yaml -d ./ # -s
# when newton_bond is on, verlet.cpp calls reverse_comm() instead of our manual call
mpirun -np 1 ${LAMMPS_ROOT}/build-test/test_pair_style manybody-pair-ani-pbc-single-cpu-newton_bond-on.yaml -d ./ # -s

if [ $NUM_GPUs -gt 0 ]; then
    mpirun -np 1 ${LAMMPS_ROOT}/build-test/test_pair_style manybody-pair-ani-pbc-single-cuda.yaml -d ./ # -s
    mpirun -np 1 ${LAMMPS_ROOT}/build-test/test_pair_style manybody-pair-ani-single-cuda.yaml -d ./ # -s
fi
