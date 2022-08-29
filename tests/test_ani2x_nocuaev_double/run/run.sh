#!/bin/bash
set -ex

mpirun -np 1 /blue/roitberg/apps/lammps/build-test/lmp_mpi -in in.lammps
mpirun -np 2 /blue/roitberg/apps/lammps/build-test/lmp_mpi -in in.lammps
mpirun -np 1 /blue/roitberg/apps/lammps/build-test/lmp_mpi -in in.lammps.cpu
mpirun -np 2 /blue/roitberg/apps/lammps/build-test/lmp_mpi -in in.lammps.cpu
