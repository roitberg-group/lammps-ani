import math

names = ["25k", "50k", "100k", "200k", "400k", "600k", "700k", "800k", "900k", "10000k", "100000k"]
num_atoms = []

for name in names:
    num_k = int(name[:-1])  # extract the numerical part
    num_atoms_unrounded = num_k * 1000  # convert k to actual numbers

    # Since one water molecule has 3 atoms, we divide the target by 3
    # and round to the nearest whole number to get the number of water molecules
    num_molecules = math.ceil(num_atoms_unrounded / 3)

    # Multiply the number of molecules by 3 to get the number of atoms
    # This might not be a multiple of 10, so find the nearest multiple of 10
    num_atoms_rounded = num_molecules * 3

    num_atoms.append(num_atoms_rounded)

print(num_atoms)
