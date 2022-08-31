#!/bin/bash
set -ex

NUM_GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# test_model
${LAMMPS_ANI_ROOT}/build/ani_csrc/test_model test_ani2x_nocuaev_double/ani2x_nocuaev_double.pt cpu
if [ $NUM_GPUs -gt 0 ]; then
    ${LAMMPS_ANI_ROOT}/build/ani_csrc/test_model test_ani2x_nocuaev_double/ani2x_nocuaev_double.pt cuda
fi

# test_ani2x_nocuaev_double
cd ./test_ani2x_nocuaev_double && ./test.sh
cd run && ./run.sh && cd ../../
