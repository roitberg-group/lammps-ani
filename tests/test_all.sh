#!/bin/bash
set -ex

NUM_GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# test_model
${LAMMPS_ANI_ROOT}/build/src/ani_csrc/test_model ani2x.pt cpu double
${LAMMPS_ANI_ROOT}/build/src/ani_csrc/test_model ani2x.pt cpu single
if [ $NUM_GPUs -gt 0 ]; then
    ${LAMMPS_ANI_ROOT}/build/src/ani_csrc/test_model ani2x.pt cuda double
    ${LAMMPS_ANI_ROOT}/build/src/ani_csrc/test_model ani2x.pt cuda single
fi

# test_ani2x_nocuaev_double
cd ./lammps-unittest/test_ani2x_nocuaev_double_half && ./test.sh
cd run && ./run.sh && cd ../../../

cd ./lammps-unittest/test_ani2x_nocuaev_single_half && ./test.sh
cd run && ./run.sh && cd ../../../

cd ./lammps-unittest/test_ani2x_cuaev_single_half && ./test.sh
cd run && ./run.sh && cd ../../../

cd ./lammps-unittest/test_ani2x_cuaev_single_full && ./test.sh
cd run && ./run.sh && cd ../../../
