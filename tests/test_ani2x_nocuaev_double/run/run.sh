#!/bin/bash
set -ex

# cuda
mpirun -np 1 ../../external/lammps/build-test/lmp_mpi -in in.lammps.cuda
mpirun -np 2 ../../external/lammps/build-test/lmp_mpi -in in.lammps.cuda
# cpu
mpirun -np 1 ../../external/lammps/build-test/lmp_mpi -in in.lammps.cpu
mpirun -np 2 ../../external/lammps/build-test/lmp_mpi -in in.lammps.cpu

rm log.lammps
rm water.final
