#!/bin/bash
set -ex

# export environment
source build-env.sh

# build base dependencies
source build-base.sh

# build lammps-ani
source build-lammps-ani.sh

# # test
# cd tests/
# ./test_all.sh
