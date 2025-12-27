# Usage:
# 0. Optionally set build args:
#    export MAKE_J_THREADS=20: number of threads for make -j
#    export OVERRIDE_KOKKOS_ARCH=Kokkos_ARCH_ADA89: override Kokkos architecture setting, e.g Kokkos_ARCH_VOLTA70, Kokkos_ARCH_ADA89
# 1. build base image, with lammps, kokkos and torchani
#     docker build --target base -t ghcr.io/roitberg-group/lammps-ani-base:latest -f Dockerfile .
# 2. build from a base image
#     docker build --target lammps-ani-build_from_base -t ghcr.io/roitberg-group/lammps-ani:latest -f Dockerfile .
# 3. build from scratch
#     docker build --target lammps-ani-build_from_scratch -t ghcr.io/roitberg-group/lammps-ani:latest -f Dockerfile .

ARG PYT_VER=25.06
# ==================== pytorch ====================
FROM nvcr.io/nvidia/pytorch:$PYT_VER-py3 AS pytorch

# ==================== base ====================
FROM pytorch AS base
# environment
ENV LAMMPS_ANI_ROOT=/lammps-ani
ENV LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
ENV LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/
ENV INSTALL_DIR=/usr/local
# CUDA_ARCH
ENV CMAKE_CUDA_ARCHITECTURES="6.0+PTX;7.0;7.5;8.0;8.9;9.0;10.0"
ENV TORCH_CUDA_ARCH_LIST=${CMAKE_CUDA_ARCHITECTURES}
# NGC PyTorch needs CXX11_ABI
ENV CXX11_ABI=1
# allow run OpenMPI as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# NGC Container forces using TF32, disable it
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
# Flags needed for CI
ARG MAKE_J_THREADS=""
ENV MAKE_J_THREADS=${MAKE_J_THREADS}
ARG OVERRIDE_KOKKOS_ARCH=""
ENV OVERRIDE_KOKKOS_ARCH=${OVERRIDE_KOKKOS_ARCH}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# install some packages
RUN apt-get update && \
    apt-get install -y libgsl-dev

# Copy files into container
COPY . $LAMMPS_ANI_ROOT
# Build base dependencies: lammps, kokkos, and torchani
RUN cd $LAMMPS_ANI_ROOT \
    && ./build-base.sh
# set work directory
WORKDIR $LAMMPS_ANI_ROOT

# ==================== lammps-ani-build_from_base ====================
FROM ghcr.io/roitberg-group/lammps-ani-base:latest AS lammps-ani-build_from_base
COPY . $LAMMPS_ANI_ROOT
RUN ./build-lammps-ani.sh

# ==================== lammps-ani-build_from_scratch ====================
FROM base AS lammps-ani-build_from_scratch
COPY . $LAMMPS_ANI_ROOT
RUN ./build-lammps-ani.sh

# Cleanup
# RUN rm -rf $LAMMPS_ANI_ROOT/.git
