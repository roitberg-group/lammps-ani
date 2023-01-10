#!/bin/bash
set -ex

NUM_MODELS=1
for NUM_MODELS in 1 8
do 
    echo ============================== $NUM_GPUS ==============================
    for NUM_GPUS in 1 2 4 8
    do
        echo ========== $NUM_MODELS ==========
        # mpirun -np ${NUM_GPUS} ${LAMMPS_ROOT}/build/lmp_mpi -var num_models ${NUM_MODELS} -log log-models_${NUM_MODELS}-gpus_${NUM_GPUS}.lammps -in in.lammps
        echo 
        LAMMPS_ANI_PROFILING=1 mpirun -np ${NUM_GPUS} ${LAMMPS_ROOT}/build/lmp_mpi -k on g ${NUM_GPUS} -sf kk -pk kokkos gpu/aware on -var newton_pair on -var num_models ${NUM_MODELS} -log log-kokkos-models_${NUM_MODELS}-gpus_${NUM_GPUS}.lammps -in in.lammps        
    done
done

