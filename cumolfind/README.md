# cumolfind - CUDA Accelerated Molecule Finder

Purpose: This package is designed to analyze extensive trajectory data and identify specific molecules of interest within that trajectory.

## Key Features

- Users can define the molecules of interest in a custom database.
- The program searches for these molecules in each frame of the trajectory. This is done by initially fragmenting the data to identify subgraphs, and then determining if these subgraphs match any molecules in the user's database.
- The identified molecule information is saved in a dataframe for easy post-analysis.

## Environment Setup on Blackwell GPUs (CUDA 12.9.1)

(Before installing on HiPerGator, purge all modules then load cuda/12.9.1)

```bash
cd /path/to/torchani
mamba create -f blackwell.yaml  # Edit environment name on line 1 of this file as you please
mamba activate ENV_NAME 
pip install --config-settings=--global-option=ext --no-build-isolation --no-deps -v .

cd /path/to/cumolfind
mamba install -c rapidsai -c conda-forge -c nvidia cudf=25.02 cugraph=25.02 cuda-version=12.9 mdtraj

pip install warp-lang

pip install -e .
```

## Environment Setup (CUDA 11)

```bash
mamba create -n rapids-23.10 -c rapidsai -c conda-forge -c nvidia \
cudf=23.10 cugraph=23.10 python=3.11 cuda-version=12.9 \
pytorch jupyterlab \
pubchempy ase mdtraj tqdm ambertools

cd /path/to/torchani_sandbox
pip install -r dev_requirements.txt
pip install --config-settings=--global-option=ext-all-sms --no-build-isolation --no-deps -v .
```

RAPIDS version 23.02 or later is required. This enables configuring PyTorch to use RAPIDS Memory Manager (RMM) for GPU memory allocation, facilitating effective memory sharing between RAPIDS and PyTorch.

## Installation

```bash
cd /path/to/cumolfind
pip install -e .
```

## Usage

`cumolfind` provides several command-line tools:

**Generating Molecule Database:**

```bash
cd data
python pubchem.py
```

This utility builds a molecule database using the PubChemPy library. The database includes columns such as "graph, formula, smiles, name, flatten_formula".

**Finding Molecules in Trajectory:**

```bash
cumolfind-molfind --help

Analyze trajectory

positional arguments:
  traj_file             Trajectory file to be analyzed
  top_file              Topology file to be analyzed
  mol_pq                Molecule database file

options:
  -h, --help            show this help message and exit
  --task TASK           "analyze_trajectory" or "track_molecules"
  --time_offset TIME_OFFSET
                        Time offset for the trajectory
  --dump_interval DUMP_INTERVAL
                        How many timesteps between frame dumps
  --timestep TIMESTEP   Timestep used in the simulation (fs)
  --output_dir OUTPUT_DIR
                        Output directory
  --num_segments NUM_SEGMENTS
                        Number of segments to divide the trajectory into
  --segment_index SEGMENT_INDEX
                        Index of the segment to analyze
```

Example

```bash
cumolfind-molfind logs-big/2023-10-13-163952.474802.dcd_split/2023-10-13-163952.474802_0.9ns.dcd data/mixture_228000.pdb ../../cumolfind/data/small_molecule_data.pq --time_offset=x --dump_interval=50 --timestep=0.25 --output_dir=test_analyze1 --num_segments=10 --segment_index=2
```

Use this command to analyze trajectory files and find molecules. It exports two files:

- `{traj_file}_formula.pq`: Database with "frame, local_frame, formula, count, time".
- `{traj_file}_molecule.pq`: Database with "frame, local_frame, hash, formula, smiles, name, atom_indices, time".

**Tracking origin of molecules:**

```bash
cumolfind-molfind --help

positional arguments:
  traj_file             Trajectory file to be analyzed
  top_file              Topology file to be analyzed
  mol_pq                Molecule database file

options:
  -h, --help            show this help message and exit
  --task "track_molecules"
  --time_offset TIME_OFFSET
                        Time offset for the trajectory
  --dump_interval DUMP_INTERVAL
                        How many timesteps between frame dumps
  --timestep TIMESTEP   Timestep used in the simulation (fs)
  --output_dir OUTPUT_DIR
                        Output directory
  --num_segments NUM_SEGMENTS
                        Number of segments to divide the trajectory into
  --segment_index SEGMENT_INDEX
                        Index of the segment to analyze
  --frame_stride FRAME_STRIDE
                        How many frames to skip when searching for mol origin
#  --frame_to_track_mol_origin
#                        Path to a .pq file that you want to find mol origin for
  
```

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

**Submit analysis job parallaly:**

```bash
python /blue/roitberg/apps/lammps-ani/cumolfind/submit_analysis.py --help
usage: submit_analysis.py [-h] --traj TRAJ --top TOP --num_segments NUM_SEGMENTS --mol_pq MOL_PQ [--output_dir OUTPUT_DIR] [-y]

Parallelize cumolfind analysis.

optional arguments:
  -h, --help            show this help message and exit
  --traj TRAJ           Directory containing trajectory files or a single trajectory file.
  --top TOP             Topology file.
  --num_segments NUM_SEGMENTS
                        Number of segments for each trajectory.
  --mol_pq MOL_PQ       Molecule database file
  --output_dir OUTPUT_DIR
                        Output directory
  -y                    If provided, the job will be submitted. If not, the job will only be prepared but not submitted.
```

Example

```bash
python /blue/roitberg/apps/lammps-ani/cumolfind/submit_analysis.py --traj=/red/roitberg/22M_20231216_testrun/ --top=/blue/roitberg/apps/lammps-ani/examples/early_earth/data/mixture_22800000.pdb --task="analyze_trajectory" --num_segments=2 --mol_pq=/blue/roitberg/apps/lammps-ani/cumolfind/data/animal_acid.pq
```
