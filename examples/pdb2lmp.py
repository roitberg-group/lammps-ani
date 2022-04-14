from ase.io import read
import os
import numpy as np
import argparse
import warnings

header = """# LAMMPS data
{num_atoms} atoms
7 atom types
{xlo} {xhi}  xlo xhi
{ylo} {yhi}  ylo yhi
{zlo} {zhi}  zlo zhi
0.0 0.0 0.0 xy xz yz

Masses

1  1.008        # H
2 12.010700     # C
3 14.0067       # N
4 15.999        # O
5 32.06         # S
6 18.998403163  # F
7 35.45         # Cl

Atoms

"""


numbers_to_species = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4, 9: 5, 17: 6}
numbers_to_lmp_types = {1: 1, 6: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7}

def generate_data(input_file, output_file, system_size=None):

    mol = read(input_file)
    cell = mol.cell
    if system_size is None or system_size > len(mol):
        num_atoms = len(mol)
    else:
        num_atoms = system_size
    xlen_half, ylen_half, zlen_half = cell.lengths() / 2

    data = ""
    data += header.format(num_atoms=num_atoms, xlo=-xlen_half, xhi=xlen_half,
                          ylo=-ylen_half, yhi=ylen_half, zlo=-zlen_half, zhi=zlen_half)

    # header
    print(f"Box information:\nlengths: {cell.lengths()}\nangles: {cell.angles()}\n")
    print(f"Assume the box is centered, you may need to modify it if necessary\n")
    if not np.array_equal(cell.angles(), [90., 90., 90.]):
        warnings.warn("This is not an orthogonal simulation box, please modify the header!")
    print(f"Generated header is the following:\n{data}")

    # atoms data
    symbols = mol.get_chemical_symbols()
    positions = mol.get_positions()
    numbers = mol.get_atomic_numbers()
    types = [numbers_to_lmp_types[i] for i in numbers]
    for i in range(num_atoms):
        position = positions[i]
        line = f"{i+1}\t{types[i]}\t{position[0]}\t{position[1]}\t{position[2]}\t# {symbols[i]}\n"
        data += (line)

    # write out data
    with open(output_file, "w") as file:
        file.write(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('--system_size', type=int, default=None)
    args = parser.parse_args()

    generate_data(args.in_file, args.out_file, args.system_size)
