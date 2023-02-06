#!/bin/bash
set -e
TIMESTAMP=`date +%F-%H%M`

# kokkos yes is only available for A100 GPU
KOKKOS_OPTION=(no)
DIR_OPTION=(20k)
NUM_MODELS_OPTION=(1)
NUM_GPUS_OPTION=(1)

# benchmark all
# KOKKOS_OPTION=(yes no)
# DIR_OPTION=(20k 300k)
# NUM_MODELS_OPTION=(1 8)
# NUM_GPUS_OPTION=(1 2 4 8)

# 20k or 300k
for RUN_KOKKOS in ${KOKKOS_OPTION[@]}; do
    for DIR in ${DIR_OPTION[@]}; do
        for NUM_MODELS in ${NUM_MODELS_OPTION[@]}; do
            for NUM_GPUS in ${NUM_GPUS_OPTION[@]}; do
                echo ============================ RUN_KOKKOS: ${RUN_KOKKOS}, DIR: ${DIR}, NUM_MODELS: ${NUM_MODELS}, NUM_GPUS: ${NUM_GPUS} ============================
                printf '============================ %s\t%s\t%s\t%s ============================\n' ${RUN_KOKKOS} ${DIR} ${NUM_MODELS} ${NUM_GPUS}
                if  [[ $RUN_KOKKOS == "yes" ]]; then
                    # run with kokkos
                    # LAMMPS_ANI_PROFILING=1 is only for profiling purpose to show the correct timing breakdown
                    (set -x;
                    LAMMPS_ANI_PROFILING=1 mpirun -np ${NUM_GPUS} ${LAMMPS_ROOT}/build/lmp_mpi \
                        -k on g ${NUM_GPUS} -sf kk -pk kokkos gpu/aware on \
                        -var newton_pair on -var num_models ${NUM_MODELS} -var datafile ${DIR}/water.data \
                        -log ${DIR}/${TIMESTAMP}-kokkos-models_${NUM_MODELS}-gpus_${NUM_GPUS}.log -in in.lammps
                    )
                else
                    # run without kokkos
                    (set -x;
                    mpirun -np ${NUM_GPUS} ${LAMMPS_ROOT}/build/lmp_mpi \
                        -var newton_pair off -var num_models ${NUM_MODELS} -var datafile ${DIR}/water.data \
                        -log ${DIR}/${TIMESTAMP}-models_${NUM_MODELS}-gpus_${NUM_GPUS}.log -in in.lammps
                    )
                fi
            done
        done
    done
done
