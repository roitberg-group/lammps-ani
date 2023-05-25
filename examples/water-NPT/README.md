## Introduction
This example demonstrates how to perform an NPT simulation of water using the LAMMPS-ANI interface.

## Background
There are two approaches to calculating stress in molecular dynamics simulations. The first approach<sup>[1]</sup> calculates stress by taking the dot product of total forces (per atom) and the displacement vector. This method is applicable to non-periodic boundary systems.

$$
P=\frac{N k_{\mathrm{B}} T}{V}+\frac{1}{3 V}\left\langle\sum_{i=1}^N \mathbf{r}_i \cdot \mathbf{f}_i\right\rangle
$$

To use this approach for a periodic boundary system, another term to account for the box lengths needs to be added:

$$
P=\frac{N k_{\mathrm{B}} T}{V}+\left\langle\frac{1}{3 V} \sum_{i=1}^N \mathbf{r}_i \cdot \mathbf{f}_i-\frac{1}{3 L^2} \frac{\partial U}{\partial L}\right\rangle
$$

This method is the one implemented in the current torchani stress calculation. https://github.com/aiqm/torchani/pull/218, https://github.com/aiqm/torchani/pull/387

The second approach (this PR) uses the partial_fdotr approach, which does not require the cell's box information and could be used in the lammps interface, where the model can only see the neighbor list (local atoms, ghost atoms) without the cell's information. The benefit of this approach is that it can be used in multiple domains.

$$
P=\frac{N k_{\mathrm{B}} T}{V}+\frac{1}{6 V}\left\langle 
\sum_{i=1}^N \sum_{j \neq i}^N r_{i j} \cdot f_{i j}
\right\rangle
$$

The approach should be universal and work for all potentials, including 2-body (radial), 3-body (angular), repulsion, dispersion, etc. This requires the very first `diff_vectors` to be saved and differentiable.

Reference:  
[1] Louwerse, M. J.; Baerends, E. J. Calculation of Pressure in Case of Periodic Boundary Conditions. Chem Phys Lett 2006, 421 (1–3), 138–141. https://doi.org/10.1016/j.cplett.2006.01.087.  
[2] Thompson, A. P.; Plimpton, S. J.; Mattson, W. General Formulation of Pressure and Stress Tensor for Arbitrary Many-Body Interaction Potentials under Periodic Boundary Conditions. J Chem Phys 2009, 131 (15), 154107. https://doi.org/10.1063/1.3245303.

## Lammps vs ASE

Run Lammps
```bash
./run_lammps.sh
# plot the volume, density, temperature and pressure
python plot_lammps.py lammps_logfile
```

Run ase
```bash
python run_ase.py
# plot the volume, density, temperature and pressure
python plot_ase.py ase_logfile
```

The results are very close.

| LAMMPS Simulation | ASE Simulation |
|:-----------------:|:--------------:|
| ![](resc/lammps.png) | ![](resc/ase.png) |

Movie of the lammps simulation in the first 5 ps:
<p align="center">
  <a href="resc/lammps.mp4">
    <img src="resc/lammps_movie.png" width="300">
  </a>
</p>

## Repulsion model
Working in progress, there is still problems, the density finally reach to 0.8 g/cm3, which is not correct.
It is possible because of dispersion is not included.


## Multi-GPU tests

## Limitatoins
currently only work with PyAEV, CUAEV does not work for stress/pressure calculation

