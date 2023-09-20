import math
import subprocess

# Define the template for the packmol input file.
template = """
tolerance 2.0
filetype pdb
output combustion-0.25-{name}.pdb

# box size is {length}.
structure methane.pdb
  number {num_mols_ch4}
  inside box 1. 1. 1. {length_1} {length_1} {length_1}
end structure

structure O2.pdb
  number {num_mols_o2}
  inside box 1. 1. 1. {length_1} {length_1} {length_1}
end structure
"""

# Simulation details.
names = ["300k",]
num_atoms = [300006,]

# CH4 + 2O2
density = 0.25  # g / cm^3

# Iterate over each simulation.
for i, atoms in enumerate(num_atoms):
    print(f"========= Simulation {names[i]} with {atoms} atoms ========")

    # Calculate the number of molecules.
    num_mols = atoms / 9
    assert num_mols.is_integer(), "Number of atoms must be a multiple of 3"
    num_mols = int(num_mols)

    # Calculate the mass of the molecules.
    mass = (num_mols * (12.011 + 4 * 1.008 + 4 * 15.999)) / 6.0221408e23  # g

    # Calculate the volume of the box.
    volume = mass / density  # cm^3
    volume *= (1e8) ** 3  # Convert volume to Angstrom^3.

    # Calculate the box length.
    box_length = math.pow(volume, 1 / 3)  # Angstrom

    print(f"Box length: {box_length} Angstrom")

    # Prepare the content for the packmol input file.
    content = template.format(
        name=names[i], length=box_length, length_1=box_length - 1, num_mols_ch4=num_mols, num_mols_o2=num_mols*2
    )

    # Write the packmol input file.
    inp_file = f"system-{names[i]}.inp"
    with open(inp_file, "w") as f:
        f.write(content)

    # Run packmol to generate the pdb file.
    subprocess.run(f"packmol < {inp_file}", check=True, shell=True)

    # Add the box length information to the generated pdb file.
    pdb_file = f"combustion-0.25-{names[i]}.pdb"
    with open(pdb_file, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        f.write(f"CRYST1{box_length:9.3f}{box_length:9.3f}{box_length:9.3f}  90.00  90.00  90.00               1\n")
        f.writelines(lines)
