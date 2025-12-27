
#!/bin/bash
set -ex

TIMESTAMP=`date +%F-%H%M%S`


################################# Configure here ################################# 

RUN_NAME=run
LAMMPS_INPUT=in.lammps
RUN_KOKKOS=yes
NUM_GPUS=1
NUM_MODELS=1
DATA_FILE=alanine-dipeptide.data
TIMESTEP=0.1
# choose ani2x.pt or ani2x_repulsion.pt
MODEL_FILE=${LAMMPS_ANI_ROOT}/tests/ani2x_repulsion.pt

# create logs folder
mkdir -p logs

#################################################################################


if  [[ $RUN_KOKKOS == "yes" ]]; then
    # run with kokkos
    # LAMMPS_ANI_PROFILING=1 is only for profiling purpose to show the correct timing breakdown
    LAMMPS_ANI_PROFILING=1 mpirun -np ${NUM_GPUS} ${LAMMPS_ROOT}/build/lmp_mpi \
        -k on g ${NUM_GPUS} -sf kk -pk kokkos gpu/aware on \
        -var newton_pair on -var num_models ${NUM_MODELS} -var datafile ${DATA_FILE} -var timestamp ${TIMESTAMP} -var modelfile ${MODEL_FILE} -var timestep ${TIMESTEP}\
        -log logs/${TIMESTAMP}-kokkos-models_${NUM_MODELS}-gpus_${NUM_GPUS}-${RUN_NAME}.log -in ${LAMMPS_INPUT}
else
    # run without kokkos
    mpirun -np ${NUM_GPUS} ${LAMMPS_ROOT}/build/lmp_mpi \
        -var newton_pair off -var num_models ${NUM_MODELS} -var datafile ${DATA_FILE} -var timestamp ${TIMESTAMP} -var modelfile ${MODEL_FILE} -var timestep ${TIMESTEP}\
        -log logs/${TIMESTAMP}-models_${NUM_MODELS}-gpus_${NUM_GPUS}-${RUN_NAME}.log -in ${LAMMPS_INPUT}
fi
