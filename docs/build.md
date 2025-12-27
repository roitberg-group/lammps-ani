# Build Instructions

## Install to ~/.local (optional)

Instead of using `source ./build-env.sh` every time, you can install LAMMPS binaries to `~/.local`:

```bash
export INSTALL_DIR=${HOME}/.local
./build.sh
```

Then add to `~/.bashrc`:
```bash
export PATH=${HOME}/.local/bin:$PATH
export LD_LIBRARY_PATH=${HOME}/.local/lib:$LD_LIBRARY_PATH
export LAMMPS_PLUGIN_PATH=${HOME}/.local/lib
```

After this, you can use `lmp_mpi` directly without sourcing build-env.sh.
