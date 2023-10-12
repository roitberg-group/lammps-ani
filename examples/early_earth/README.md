# Early Earth Simulation

This document provides instructions for running a molecular dynamics (MD) simulation of the Miller's Early Earth experiment[^1] using LAMMPS-ANI with ANI1x_NR[^2] model.

### Model
ANI1x_NR[^2] model: https://github.com/atomistic-ml/ani-1xnr


### Procedure

Following the simulation steps in Reference[^2]:

- Packmol was utilized to randomly place 16 H2 , 14 H2O, 14 CO, 14 NH3 and 14 CH4 in a cubic simulation box with edge lengths of 12.1 Å, resulting in a density of 1.067 g/cc. (I will use the structure file found from [github.com/leeping/nanoreactor/example.md/start.xyz](https://github.com/leeping/nanoreactor/blob/e069fffb93692d3ed04cb73b9a7c6a8fa9bce3ca/example.md/start.xyz))
- The simulation was run with Langevin dynamics for over 4 ns with a time step of 0.25 fs. 
  - The temperature was linearly increased from 0 K to 300 K in the first 100 ps.
  - Then, the temperature was linearly increased from 300 K to 2500 K in the next 100 ps. 
  - The temperature was then maintained at 2500 K for 4000 ps. 
  - The system was then cooled from 2500 K to 300 K over the final 200 ps. 
  - Snapshots and properties were recorded every 12.5 fs (50 time steps)

I Tried to run NPT first. When using `ani1x_nr_repulsion.pt` model, the density quickly went down from 1.04 g/cc to 0.4 g/cc after 5000 steps (1.25ps). this is not solvent, so maybe NPT is not necessary?

```bash
python run_one.py start.data --kokkos --num_gpus=1 --input_file=in.npt.lammps --log_dir=logs --ani_model_file='ani1x_nr_repulsion.pt' --run_name=early_earth_npt --ani_num_models=1 --timestep=0.25 --run_steps=40000 --run
```

I used all the models in the ensemble for the simulation.
```bash
python run_one.py start.data --kokkos --num_gpus=1 --input_file=in.lammps --log_dir=logs --ani_model_file='ani1x_nr_repulsion.pt' --run_name=early_earth_ani1x_nr_repulsion --ani_num_models=-1 --timestep=0.25 --run

python run_one.py start.data --kokkos --num_gpus=1 --input_file=in.lammps --log_dir=logs --ani_model_file='ani1x_nr.pt' --run_name=early_earth_ani1x_nr --ani_num_models=-1 --timestep=0.25 --run

python run_one.py start.data --kokkos --num_gpus=1 --input_file=in.lammps --log_dir=logs --ani_model_file='ani2x_repulsion.pt' --run_name=early_earth_ani2x_repulsion --ani_num_models=-1 --timestep=0.25 --run

python run_one.py mixture_228000.data --kokkos --num_gpus=1 --input_file=in.lammps --log_dir=logs --ani_model_file='ani1x_nr.pt' --run_name=scale_early_earth_ani1x_nr --ani_num_models=-1 --timestep=0.25 --run
```

[^1]: Miller, S. L. A Production of Amino Acids Under Possible Primitive Earth Conditions. Science 1953, 117 (3046), 528–529. https://doi.org/10.1126/science.117.3046.528.
[^2]: ZHANG, S.; Makoś, M.; Jadrich, R.; Kraka, E.; Barros, K.; Nebgen, B.; Tretiak, S.; Isayev, O.; Lubbers, N.; Messerly, R.; Smith, J. Exploring the Frontiers of Chemistry with a General Reactive Machine Learning Potential. 2022. https://doi.org/10.26434/chemrxiv-2022-15ct6-v2.


### Generate starting geometries
Please check folder [generate_big_start_system](generate_big_start_system/).

Some note to keep in mind: althought the box size is 12.1 Å, we need to set it as 11.1 Å at packmol to make some buffer space for the periodic boundary condition.
```
inside box 1.0 1.0 1.0 11.1 11.1 11.1
```

Same for the larger system (228000 atoms), increase the number of each molecule by 1000 times. To maintain the same density, you would need to scale the volume of the box cubically,  cube root of 1000 is 10, so the box size should be 121.0 Å. Subtract 1.0 Å for the buffer space.
```
inside box 1.0 1.0 1.0 120.0 120.0 120.0
```

### Extracting frames from the trajectory

For the ani1x_nr model, the interesting frames are from 190070 to 190088. We can extract them using the following command:
```bash
# use 190070 frame as the topology file
python ../combustion/analyze.py start.pdb logs/2023-08-02-035928.488223.dcd --frame=190070
# the following extract 200 frames and save as a new dcd trajectory
python ../combustion/analyze.py start.pdb logs/2023-08-02-035928.488223.dcd --frame=190000 --frame-end=190200
```

Run for the larger system (228000 atoms)

```bash
python run_one.py mixture_228000.data --kokkos --num_gpus=8 --input_file=in.big.lammps --log_dir=logs-big --ani_model_file='ani1x_nr.pt' --run_name=scale_early_earth_ani1x_nr --ani_num_models=-1 --timestep=0.25 --run
# 46,14440,C2H5NO2,9,0.1805

# extract a single frame
python ../combustion/analyze.py mixture_228000.pdb logs/2023-08-28-022042.908705.dcd --frame=14440
# extract a single frame
python ../combustion/analyze.py mixture_228000.pdb logs/2023-08-28-022042.908705.dcd --frame=14440 --frame_end=14450
# identify glycines from all C2H5NO2 molecules
python ../combustion/analyze.py mixture_228000.pdb logs-big/2023-08-28-022042.908705.dcd --csv_file=analyze/2023-08-28-022042.908705.csv
```
