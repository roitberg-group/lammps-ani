ARG PYT_VER=22.08
FROM nvcr.io/nvidia/pytorch:$PYT_VER-py3

# environment
ENV LAMMPS_ANI_ROOT=/lammps-ani
ENV LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
ENV LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/
# CUDA_ARCH
ENV CMAKE_CUDA_ARCHITECTURES="6.0+PTX;7.5;8.0"
ENV TORCH_CUDA_ARCH_LIST="6.0+PTX;7.5;8.0"
# NGC PyTorch needs CXX11_ABI
ENV CXX11_ABI=1
# allow run OpenMPI as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Copy files into container
COPY . $LAMMPS_ANI_ROOT

# Install modulus and dependencies
RUN cd $LAMMPS_ANI_ROOT \
    && ./build.sh

# Cleanup
RUN rm -rf $LAMMPS_ANI_ROOT/.git

WORKDIR $LAMMPS_ANI_ROOT
