#!/bin/bash
et -ex

mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -in in.lammps -var num_models 1
mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -in in.lammps -var num_models 4
mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -in in.lammps -var num_models 8
mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -k on g 1 -sf kk -pk kokkos newton on -in in.lammps -var num_models 1
mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -k on g 1 -sf kk -pk kokkos newton on -in in.lammps -var num_models 4
mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -k on g 1 -sf kk -pk kokkos newton on -in in.lammps -var num_models 8
