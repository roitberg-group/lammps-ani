# LAMMPS-ANI
A plugin to run torchani on LAMMPS.  
On hipergator, the compiled program and a working example script could be found at `/blue/roitberg/apps/lammps-ani/examples/water/submit.sh`

## Requirement
Run an interactive session
```
srun --qos=roitberg --account=roitberg --nodes=1 --ntasks=2 --cpus-per-task=2 --mem=20gb --gres=gpu:2 --partition=hpg-ai -t 10:00:00 --pty /bin/bash -i
module load cuda/11.4.3 gcc/9.3.0 openmpi/4.0.5 cmake/3.21.3 git/2.30.1 singularity
```

pytorch and cudnn
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge cudnn=8.3.2
```

## Singularity Usage
You could use the pre-built Singularity container to avoid build yourself.

```bash
git clone --recursive git@github.com:roitberg-group/lammps-ani.git
singularity pull -F docker://ghcr.io/roitberg-group/lammps-ani:master
mkdir -p ~/singularity-home
# exec into container
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES singularity exec --cleanenv -H ~/singularity-home:/home --nv lammps-ani_master.sif /bin/bash
# test
cd lammps-ani
nvidia-smi && cd external/torchani_sandbox && python setup.py install --ext --user && cd ../../ && cd tests/ && python save_ani_nocuaev.py && ./test_all.sh
```

## Build
```bash
./build.sh
```

## Run example
make sure `LAMMPS_PLUGIN_PATH` and `LAMMPS_ROOT` are set correctly
```
export LAMMPS_PLUGIN_PATH=/blue/roitberg/apps/lammps-ani/build/
cd examples/water/
mpirun -np 8 ${LAMMPS_ROOT}/build/lmp_mpi -in in.lammps
```
