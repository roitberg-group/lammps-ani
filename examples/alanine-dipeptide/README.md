# Alanine Dipeptide Simulation

This document provides instructions for running a molecular dynamics (MD) simulation of the alanine dipeptide using LAMMPS. The simulation can be performed with or without the SHAKE algorithm to constrain bond lengths.

### 1. Generate LAMMPS Data File

Generate the LAMMPS data file from the provided PDB file (alanine-dipeptide.pdb) using the pdb2lmp.py script. The resulting data file will be named alanine-dipeptide.data.

```bash
python pdb2lmp.py alanine-dipeptide.pdb alanine-dipeptide.data
```

### 2. Equilibrate the System with NPT

Prior to the production simulation, it's necessary to relax the system to attain the correct density through an NPT simulation. Refer to the [water-NPT](../water-NPT/) example for further details about NPT simulations. For systems without bonds information (which will be used for simulations without the SHAKE algorithm), execute `run_npt.sh`. This script uses `alanine-dipeptide.data` as the initial structure and saves the final equilibrated structure as `alanine-dipeptide.npt.data`.

### 3. Run Simulation Without SHAKE

To run the simulation without SHAKE constraints, you can adjust the timestep by setting the TIMESTEP variable in the run.sh script. Then, execute the script to start the simulation:

```bash
./run.sh
```

### 4. Run Simulation With SHAKE Constraints

For simulations involving SHAKE constraints, a LAMMPS data file inclusive of bond information and bond coefficients needs to be generated. Use the --bonds option to specify the bond types (OH, CH, NH):
```bash
python pdb2lmp.py alanine-dipeptide.pdb alanine-dipeptide-bonds.data --bonds OH,CH,NH
```

Equilibration of the system through an NPT simulation is also required in this case. Execute `run_shake_npt.sh`, which employs `alanine-dipeptide-bonds.data` as the initial structure and outputs the final equilibrated structure as `alanine-dipeptide-bonds.npt.data`. For clarity, note that the SHAKE algorithm is not used during this NPT equilibration stage, as SHAKE is incompatible with simulations that change the box size, such as NPT.

Following these preparatory steps, you're now ready to run the simulation:

```bash
./run_shake.sh
```

The LAMMPS input file for the SHAKE simulation is `in.shake.lammps`. Here are the key differences between `in.shake.lammps` and the `in.lammps`:

- The `atom_style` is set to `bond`, and `bond_style zero` is added to specify zero-energy bonds for SHAKE.
- The `special_bonds` command is added to include 1-2, 1-3, and 1-4 bonded pairs in the non-bonded potential neighbor lists.
- A `fix` command is added to apply the SHAKE algorithm to constrain the specified bond types.

The SHAKE algorithm allows for a larger timestep (e.g., 2 fs) without compromising stability. Without SHAKE, the simulation may fail quickly due to high-frequency bond vibrations.

### 5. Analyze Results

After the simulation is complete, you can analyze the results. Key aspects to examine include:

Temperature: Monitor the temperature of the system over time to assess its stability and equilibration.

Ramachandran Plot: Generate a Ramachandran plot to visualize the distribution of the backbone dihedral angles (phi and psi) and assess the conformational states visited by the alanine dipeptide.

For the Ramachandran plot, you can use computational tools such as MDTraj or PyMOL to calculate the phi and psi angles from the trajectory and create the plot.

### 6. Conclusion

This document provides the necessary steps to run an MD simulation of the alanine dipeptide using LAMMPS, both with and without SHAKE constraints. The results can be analyzed to study the conformational dynamics of this model peptide system.
