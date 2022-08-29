#!/bin/bash
set -ex

# cuda
mpirun -np 1 /blue/roitberg/apps/lammps/build-test/lmp_mpi -in in.lammps.cuda
mpirun -np 2 /blue/roitberg/apps/lammps/build-test/lmp_mpi -in in.lammps.cuda
# cpu
mpirun -np 1 /blue/roitberg/apps/lammps/build-test/lmp_mpi -in in.lammps.cpu
mpirun -np 2 /blue/roitberg/apps/lammps/build-test/lmp_mpi -in in.lammps.cpu

rm log.lammps
rm water.final
