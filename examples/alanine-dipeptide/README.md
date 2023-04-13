# Alanine Dipeptide Simulation

This document provides instructions for running a molecular dynamics (MD) simulation of the alanine dipeptide using LAMMPS. The simulation can be performed with or without the SHAKE algorithm to constrain bond lengths.

### 1. Generate LAMMPS Data File
Generate the LAMMPS data file from the provided PDB file (alanine-dipeptide.pdb) using the pdb2lmp.py script. The resulting data file will be named alanine-dipeptide.data.

```bash
python pdb2lmp.py alanine-dipeptide.pdb alanine-dipeptide.data
```

### 2. Run Simulation Without SHAKE
To run the simulation without SHAKE constraints, you can adjust the timestep by setting the TIMESTEP variable in the run.sh script. Then, execute the script to start the simulation:

```bash
./run.sh
```

### 3. Run Simulation With SHAKE Constraints
For simulations with SHAKE constraints, we need to generate a LAMMPS data file that includes bond information and bond coefficients. The `--bonds` option specifies the bond types (OH, CH, NH) to include:

```bash
python pdb2lmp.py alanine-dipeptide.pdb alanine-dipeptide-bonds.data --bonds OH,CH,NH
```

The LAMMPS input file for the SHAKE simulation is `in.shake.lammps`. Here are the key differences between `in.shake.lammps` and the `in.lammps`:

- The `atom_style` is set to `bond`, and `bond_style zero` is added to specify zero-energy bonds for SHAKE.
- The `special_bonds` command is added to include 1-2, 1-3, and 1-4 bonded pairs in the non-bonded potential neighbor lists.
- A `fix` command is added to apply the SHAKE algorithm to constrain the specified bond types.

The SHAKE algorithm allows for a larger timestep (e.g., 2 fs) without compromising stability. Without SHAKE, the simulation may fail quickly due to high-frequency bond vibrations.

### 4. Analyze Results
After the simulation is complete, you can analyze the results. Key aspects to examine include:

Temperature: Monitor the temperature of the system over time to assess its stability and equilibration.
Ramachandran Plot: Generate a Ramachandran plot to visualize the distribution of the backbone dihedral angles (phi and psi) and assess the conformational states visited by the alanine dipeptide.
For the Ramachandran plot, you can use computational tools such as MDTraj or PyMOL to calculate the phi and psi angles from the trajectory and create the plot.

### 5. Conclusion
This document provides the necessary steps to run an MD simulation of the alanine dipeptide using LAMMPS, both with and without SHAKE constraints. The results can be analyzed to study the conformational dynamics of this model peptide system.
