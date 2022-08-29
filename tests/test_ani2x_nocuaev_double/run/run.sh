#!/bin/bash
set -ex

mpirun -np 1 ../../../external/lammps/build-test/lmp_mpi -in in.lammps.cuda
mpirun -np 1 ../../../external/lammps/build-test/lmp_mpi -in in.lammps.cpu

ntasks=${SLURM_NTASKS:-1}
if [ $ntasks -gt 1 ]; then
    mpirun -np 2 ../../../external/lammps/build-test/lmp_mpi -in in.lammps.cuda
    mpirun -np 2 ../../../external/lammps/build-test/lmp_mpi -in in.lammps.cpu
fi

rm log.lammps
rm water.final
