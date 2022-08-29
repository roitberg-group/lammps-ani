#!/bin/bash
set -ex

# test_model
../build/ani_csrc/test_model test_ani2x_nocuaev_double/ani2x_nocuaev_double.pt cpu
../build/ani_csrc/test_model test_ani2x_nocuaev_double/ani2x_nocuaev_double.pt cuda

# test_ani2x_nocuaev_double
cd ./test_ani2x_nocuaev_double && ./test.sh
cd run && ./run.sh && cd ../../
