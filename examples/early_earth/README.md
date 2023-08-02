# Early Earth Simulation

This document provides instructions for running a molecular dynamics (MD) simulation of the Miller's Early Earth experiment[^1] using LAMMPS-ANI with ANI1x_NR[^2] model.

### Model
ANI1x_NR[^2] model: https://github.com/atomistic-ml/ani-1xnr


### Procedure

Following the simulation steps in Reference[^2]:

- Packmol was utilized to randomly place 16 H2 , 14 H2 O, 14 CO, 14 NH3 and 14 CH4 in a cubic simulation box with edge lengths of 12.1 Å, resulting in a density of 1.067 g/cc.
- The simulation was run with Langevin dynamics for over 4 ns with a time step of 0.25 fs. 
  - The temperature was linearly increased from 0 K to 300 K in the first 100 ps. 
  - Then, the temperature was linearly increased from 300 K to 2500 K in the next 100 ps. 
  - The temperature was then maintained at 2500 K for 4000 ps. 
  - The system was then cooled from 2500 K to 300 K over the final 200 ps. 
  - Snapshots and properties were recorded every 12.5 fs (50 time steps)


[^1]: Miller, S. L. A Production of Amino Acids Under Possible Primitive Earth Conditions. Science 1953, 117 (3046), 528–529. https://doi.org/10.1126/science.117.3046.528.
[^2]: ZHANG, S.; Makoś, M.; Jadrich, R.; Kraka, E.; Barros, K.; Nebgen, B.; Tretiak, S.; Isayev, O.; Lubbers, N.; Messerly, R.; Smith, J. Exploring the Frontiers of Chemistry with a General Reactive Machine Learning Potential. 2022. https://doi.org/10.26434/chemrxiv-2022-15ct6-v2.
