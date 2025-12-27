# Build Instructions

## Build Docker Container

```bash
# Set build args (adjust threads and GPU architecture as needed)
export DOCKER_BUILD_ARGS="--build-arg MAKE_J_THREADS=20 --build-arg MAX_JOBS=20 --build-arg OVERRIDE_KOKKOS_ARCH=Kokkos_ARCH_ADA89"

# Build base image (lammps, kokkos, torchani)
docker build --progress=plain $DOCKER_BUILD_ARGS --target base -t ghcr.io/roitberg-group/lammps-ani-base:latest -f Dockerfile .

# Build lammps-ani from base
docker build --progress=plain $DOCKER_BUILD_ARGS --target lammps-ani-build_from_base -t ghcr.io/roitberg-group/lammps-ani:latest -f Dockerfile .

# Push to GitHub Container Registry
docker push ghcr.io/roitberg-group/lammps-ani-base:latest
docker push ghcr.io/roitberg-group/lammps-ani:latest
```

Kokkos architecture options: `Kokkos_ARCH_VOLTA70`, `Kokkos_ARCH_AMPERE80`, `Kokkos_ARCH_ADA89`, `Kokkos_ARCH_HOPPER90`, etc.

## Install to ~/.local (optional)

By default, use `source ./build-env.sh` before running.

To install system-wide instead, set `export INSTALL_DIR=${HOME}/.local` before running `./build.sh`. Then add the following to `~/.bashrc`, so you can use `lmp_mpi` directly without sourcing build-env.sh.
```bash
export PATH=${HOME}/.local/bin:$PATH
export LD_LIBRARY_PATH=${HOME}/.local/lib:$LD_LIBRARY_PATH
export LAMMPS_PLUGIN_PATH=${HOME}/.local/lib
```
