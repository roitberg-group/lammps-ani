#!/bin/bash
set -e

# =============================================================================
# Configuration - edit these values as needed
# =============================================================================
KOKKOS_ARCH="Kokkos_ARCH_ADA89"  # VOLTA70, AMPERE80, ADA89, HOPPER90
MAKE_J_THREADS=20
REGISTRY="ghcr.io/roitberg-group"

# Extract short arch name for tagging (e.g., Kokkos_ARCH_ADA89 -> ada89)
ARCH_SHORT=$(echo "$KOKKOS_ARCH" | sed 's/Kokkos_ARCH_//' | tr '[:upper:]' '[:lower:]')

echo "Building: $KOKKOS_ARCH ($ARCH_SHORT), threads=$MAKE_J_THREADS"

DOCKER_BUILD_ARGS="--build-arg MAKE_J_THREADS=$MAKE_J_THREADS --build-arg MAX_JOBS=$MAKE_J_THREADS --build-arg OVERRIDE_KOKKOS_ARCH=$KOKKOS_ARCH"

# Build base image
docker build --progress=plain $DOCKER_BUILD_ARGS --target base -t ${REGISTRY}/lammps-ani-base:latest-${ARCH_SHORT} -f Dockerfile .

# Build lammps-ani image
docker build --progress=plain $DOCKER_BUILD_ARGS --target lammps-ani-build_from_base -t ${REGISTRY}/lammps-ani:latest-${ARCH_SHORT} -f Dockerfile .

# Push
docker push ${REGISTRY}/lammps-ani-base:latest-${ARCH_SHORT}
docker push ${REGISTRY}/lammps-ani:latest-${ARCH_SHORT}

echo "Done! Pushed:"
echo "  ${REGISTRY}/lammps-ani-base:latest-${ARCH_SHORT}"
echo "  ${REGISTRY}/lammps-ani:latest-${ARCH_SHORT}"
