#!/bin/bash
set -ex

mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -var num_models 1 -in in.lammps
mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -var num_models 4 -in in.lammps
mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -var num_models 8 -in in.lammps
mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -k on g 1 -sf kk -var newton_pair on -var num_models 1 -in in.lammps
mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -k on g 1 -sf kk -var newton_pair on -var num_models 4 -in in.lammps
mpirun -np 1 ${LAMMPS_ROOT}/build/lmp_mpi -k on g 1 -sf kk -var newton_pair on -var num_models 8 -in in.lammps
