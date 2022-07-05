#!/bin/bash

mpirun -np 1 /blue/roitberg/apps/lammps/build-test/test_pair_style manybody-pair-ani-double.yaml -d ./ -s
mpirun -np 1 /blue/roitberg/apps/lammps/build-test/test_pair_style manybody-pair-ani-pbc-double.yaml -d ./ -s
