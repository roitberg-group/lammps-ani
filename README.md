# LAMMPS-ANI
A plugin that enables LAMMPS to run molecular dynamics simulations using the TorchANI neural network potential.

## Requirement
This section describes the steps required to set up the environment for running LAMMPS-ANI.

### For HiPerGator Users
To run an interactive session on HiPerGator and load the necessary modules, use the following commands:
```bash
# Run an interactive session
srun --qos=roitberg --account=roitberg --nodes=1 --ntasks=2 --cpus-per-task=2 --mem=80gb --gres=gpu:2 --partition=hpg-ai -t 3:00:00 --pty /bin/bash -i

# Load modules
module load cuda/11.4.3 gcc/9.3.0 openmpi/4.0.5 cmake/3.21.3 git/2.30.1 netcdf/4.7.2 singularity
```

You could skip the following if you use docker or singularity container.

Next, install PyTorch and cuDNN using Conda.
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c conda-forge cudnn=8.3.2
```

NetCDF should be available in most HPC systems and can be loaded using the module load command. If you are running Ubuntu and need to install NetCDF, use the following command:
```bash
apt install libnetcdf-dev
```


### For Expanse Users

Expanse does not have the latest CUDA libraries required to compile LAMMPS-ANI. Please use the Singularity container instead. The instructions for using Singularity on Expanse are as follows:

```bash
# Request an interactive session on Expanse
srun -p gpu-shared --nodes=1 --ntasks=2 --account=[YOUR_ACCOUNT_NAME] --cpus-per-task=2 --gpus=2 --time=03:00:00 --mem=80gb --pty -u bash -i

# Load the Singularity module
module load singularitypro
```

For detailed instructions on how to use the Singularity container, please refer to the [Singularity container](#singularity-container) section.

## Docker Container
You can use the pre-built [docker container](https://github.com/roitberg-group/lammps-ani/pkgs/container/lammps-ani) to avoid manually compiling the program. Note that the pre-built container only supports Kokkos for A100 GPUs.

```bash
# Pull the Docker container
docker pull ghcr.io/roitberg-group/lammps-ani:master
# Run the Docker container
docker run --gpus all -it ghcr.io/roitberg-group/lammps-ani:master
```

## Singularity Container
Some HPC systems provide Singularity instead of Docker. The following instructions demonstrate how to use Singularity for running LAMMPS-ANI:
``` bash
# First, load the Singularity module
module load singularity  # or `module load singularitypro` for Expanse

# Set up cache and temporary directories for Singularity (required for Expanse users)
export SINGULARITY_CACHEDIR=/scratch/$USER/job_$SLURM_JOB_ID
export SINGULARITY_TMPDIR=/scratch/$USER/job_$SLURM_JOB_ID

# Create and enter a new folder
mkdir ani; cd ani

# Pull the Singularity container
singularity pull -F docker://ghcr.io/roitberg-group/lammps-ani:master

# Clone the LAMMPS-ANI repository
git clone --recursive git@github.com:roitberg-group/lammps-ani.git
```

The folder structure should now be as follows:
```bash
.
├── lammps-ani
└── lammps-ani_master.sif
```

Execute into the Singularity container:

```bash
# [TODO] slurm environment might also be needed
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES singularity exec --cleanenv -H ./:/home --nv lammps-ani_master.sif /bin/bash
# [TODO] how to run with 2 GPUs
```
The above command allows you to execute a Singularity container (lammps-ani_master.sif) on a system with NVIDIA GPUs. Let's break down the command and its components:

- `SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES`: This part sets the `CUDA_VISIBLE_DEVICES` environment variable inside the Singularity container to match the value of the `CUDA_VISIBLE_DEVICES` variable outside the container. This ensures that the container has access to the same GPUs as the host system.
- `--cleanenv`: This flag clears the environment inside the container, ensuring that only essential Singularity environment variables are set. This helps prevent conflicts between the host system and container environments.
- `-H ./:/home`: This option allows you to bind a directory from the host system (`./`, which is the current directory) to a directory inside the container (`/home`). Any changes made to the `/home` directory inside the container will be reflected in the current directory on the host system.

It's important to note that the LAMMPS-ANI plugin is installed in the `/lammps-ani `directory inside the container. However, Singularity containers are read-only by default, which means you cannot write or modify files within the container itself. To work around this limitation, you can use bind mounts (as done with the `-H ./:/home` option) to map writable directories from the host system into the container. When running simulations, make sure to use directories that are writable on the host system.

You can then run the water example:
```bash
cd lammps-ani/examples/water
# If you don't have A100 GPUs, you need to change the `RUN_KOKKOS` variable to `no` in the `run.sh` script.

# Run the example
./run.sh
```

You can also build Kokkos for the GPUs you have, which moves all the LAMMPS internal computations, such as neighbor list calculation and position update, to GPUs. It also eliminates the need for CPU-GPU data transfer and significantly improves performance. Please refer to the [Build Within Container](#build-within-container) section for more details.

The [Alanine Dipeptide](examples/alanine-dipeptide) example is also available for reference.

## Usage
To use the LAMMPS-ANI plugin, you need to specify the `pair_style` and `pair_coeff` commands in your LAMMPS input script as follows:

```bash
pair_style     ani 5.1 model_file device num_models ani_aev ani_neighbor
pair_coeff     * *
```

1. `model_file` - Path to the TorchANI model file
2. `device` - Specifies the device to use: `cuda` for GPU or `cpu` for CPU.
3. `num_models` (Optional) - Number of models to use in the ensemble. Default is `-1` to use all models.
4. `ani_aev` (Optional) - AEV computation method: `cuaev` (CUDA AEV) or `pyaev` (PyTorch AEV). Default is `cuaev`.
5. `ani_neighbor` (Optional) - Neighbor list type: `full` or `half`. Default is `full` and is prefered for performance benefit.
6. `ani_precision` (Optional) - Precision mode: `single` or `double`. Default is `single`.


## Models

Three models are currently available: `ani2x.pt`, `ani2x_repulsion.pt`, `and ani2x_ext0_repulsion.pt`. These models are defined in the [models.py](tests/models.py) file and can be tested using [test_models.py](tests/test_models.py). To use custom models, ensure that they follow the specified format, as shown in [models.py](tests/models.py).

The `ani2x_repulsion.pt` could only be exported within UF network. But all three models could be found in the container directory `/lammps-ani/tests/*.pt`.

To export the models:
```bash
cd lammps-ani
# When using singularity container, you may encounter write permission error, you could solve it by install torchani_sandbox to your home directory by:
# cd external/torchani_sandbox && python setup.py install --ext --user && cd ../../

# Save models and tests
cd tests/
pytest test_models.py -s -v

# (Optional) Simulation tests against to ASE
pytest test_lmp_with_ase.py -s -v
```

## Build from Source

You can build LAMMPS-ANI from source using the following commands:
```bash
git clone --recursive git@github.com:roitberg-group/lammps-ani.git
./build.sh
# By default, this builds CUAEV for all GPU architectures. To speed up the process, # specify CMAKE_CUDA_ARCHITECTURES for specific architectures. For example:
# CMAKE_CUDA_ARCHITECTURES="7.5;8.0" ./build.sh
```

To use the LAMMPS-ANI plugin, you need to set environment variables for the paths. You can add these variables to your `~/.bashrc` file or use the provided source ./build-env.sh script to load them automatically:

```bash
export LAMMPS_ANI_ROOT=<PATH_TO_LAMMPS_ANI_ROOT>
export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/
export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/
```
Then lammps could be envoked by
```
mpirun -np 1 ${LAMMPS_ROOT}/lmp_mpi -help
```

(Optional) User could also install lammps binaries and libraries into `${HOME}/.local`. In this case, you need to set `PATH` and `LD_LIBRARY_PATH` environment variables to be able to use `lmp_mpi` directly.
```bash
export PATH=${HOME}/.local/bin:$PATH
export LD_LIBRARY_PATH=${HOME}/.local/lib:$LD_LIBRARY_PATH
export LAMMPS_PLUGIN_PATH=${HOME}/.local/lib
```
Then lammps could be envoked by
```
mpirun -np 1 lmp_mpi -help
```

## Build within Container
[TODO] Detailed instructions for building LAMMPS-ANI within a Docker or Singularity container will be provided here.


## Run Examples

The LAMMPS-ANI plugin provides several simulation [examples](examples/) that can be found in the examples folder. To successfully run these examples, you need to properly set the `LAMMPS_ANI_ROOT`, `LAMMPS_ROOT`, and `LAMMPS_PLUGIN_PATH` environment variables. These variables are essential for locating the LAMMPS-ANI root directory, the LAMMPS root directory, and the LAMMPS plugin path, respectively.

For users running the simulations within a Docker or Singularity container, these environment variables are automatically set within the container.

For users who have built LAMMPS-ANI locally, it is important to set these environment variables manually in the `~/.bashrc` file and also in any script file used to submit SLURM jobs. You can conveniently set these variables by running the `source ./build-env.sh` script, which is provided with LAMMPS-ANI.

Here is an example of how to set these environment variables:

```bash
export LAMMPS_ANI_ROOT=<PATH_TO_LAMMPS_ANI_ROOT>  # Replace with the actual path to LAMMPS-ANI root directory
export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/  # Path to the LAMMPS root directory
export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/  # Path to the LAMMPS plugin directory
```

After setting the environment variables correctly, you can navigate to the [examples](examples/) folder and follow the instructions provided in each example to run the simulations.

## Benchmark

You can use the pre-built Docker container to run benchmarks for LAMMPS-ANI. Please note that the pre-built container supports Kokkos only for A100 GPUs.

```bash
# Pull the Docker container
docker pull ghcr.io/roitberg-group/lammps-ani:master
# Run the Docker container
docker run --gpus all -it ghcr.io/roitberg-group/lammps-ani:master
# Go to the water example directory
cd /lammps-ani/examples/water/
# Run the benchmark script (Note: For this container, Kokkos is only available for A100 GPUs)
bash benchmark.sh
```
