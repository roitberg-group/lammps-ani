# cumolfind - CUDA Accelerated Molecule Finder

Purpose: This package is designed to analyze extensive trajectory data and identify specific molecules of interest within that trajectory.

## Key Features:

- Users can define the molecules of interest in a custom database.
- The program searches for these molecules in each frame of the trajectory. This is done by initially fragmenting the data to identify subgraphs, and then determining if these subgraphs match any molecules in the user's database.
- The identified molecule information is saved in a dataframe for easy post-analysis.

## Environment Setup:

```bash
mamba create -n rapids-23.10 -c rapidsai -c conda-forge -c nvidia \
cudf=23.10 cugraph=23.10 python=3.10 cuda-version=11.8 \
pytorch jupyterlab
```

RAPIDS version 23.02 or later is required. This enables configuring PyTorch to use RAPIDS Memory Manager (RMM) for GPU memory allocation, facilitating effective memory sharing between RAPIDS and PyTorch.


## Installation
```bash
pip install -e .
```

## Usage

`cumolfind` provides several command-line tools:

**Generating Molecule Database:**

```bash
cumolfind-pubchem
```

This utility builds a molecule database using the PubChemPy library. The database includes columns such as "fragment object, hash, formula, smiles, name".

**Finding Molecules in Trajectory:**

```bash
cumolfind-molfind --traj_file [path/to/traj_file] --top_file [path/to/top_file] [other arguments]
```

Use this command to analyze trajectory files and find molecules. It exports two files:

- `{traj_file}_formula.pq`: Database with "frame, local_frame, formula, count, time".
- `{traj_file}_molecule.pq`: Database with "frame, local_frame, hash, formula, smiles, name, atom_indices, time".

**Extracting Frames:**

```bash
cumolfind-extract --traj_file [path/to/traj_file] --top_file [path/to/top_file] --mol_pq_file [path/to/molecule_file] --local_frame [frame_number]
```

This command extracts specified frames from a trajectory and exports them as PDB files.

**Splitting Trajectory:**

```bash
cumolfind-split_traj --traj_file [path/to/traj_file] [other arguments]
```

This command splits a large trajectory file into smaller segments, naming each segment with the suffix `traj_name_x.xns.dcd`, where `x.x` represents the time offset for the start of each segment.

