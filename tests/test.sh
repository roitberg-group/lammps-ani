#!/bin/bash
set -ex

mpirun -np 1 /blue/roitberg/apps/lammps/build-test/test_pair_style manybody-pair-ani-double-cpu.yaml -d ./ # -s
mpirun -np 1 /blue/roitberg/apps/lammps/build-test/test_pair_style manybody-pair-ani-double-cuda.yaml -d ./ # -s
mpirun -np 1 /blue/roitberg/apps/lammps/build-test/test_pair_style manybody-pair-ani-pbc-double-cpu.yaml -d ./ # -s
# when newton_bond is on, verlet.cpp calls reverse_comm() instead of our manual call
mpirun -np 1 /blue/roitberg/apps/lammps/build-test/test_pair_style manybody-pair-ani-pbc-double-cpu-newton_bond-on.yaml -d ./ # -s
mpirun -np 1 /blue/roitberg/apps/lammps/build-test/test_pair_style manybody-pair-ani-pbc-double-cuda.yaml -d ./ # -s

