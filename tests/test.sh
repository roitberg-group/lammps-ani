#!/bin/bash
set -ex

mpirun -np 1 /blue/roitberg/apps/lammps/build-test/test_pair_style manybody-pair-ani-double-cpu.yaml -d ./ # -s
mpirun -np 1 /blue/roitberg/apps/lammps/build-test/test_pair_style manybody-pair-ani-double-cuda.yaml -d ./ # -s
mpirun -np 1 /blue/roitberg/apps/lammps/build-test/test_pair_style manybody-pair-ani-pbc-double-cpu.yaml -d ./ # -s
mpirun -np 1 /blue/roitberg/apps/lammps/build-test/test_pair_style manybody-pair-ani-pbc-double-cuda.yaml -d ./ # -s

