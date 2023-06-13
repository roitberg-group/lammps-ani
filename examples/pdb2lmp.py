import numpy as np
import argparse
import warnings
import textwrap
from ase.io import read
from ase.geometry.analysis import Analysis

# Template for LAMMPS data file header
HEADER = """# LAMMPS data
{num_atoms} atoms
7 atom types

{num_bonds} bonds
{num_bond_types} bond types  # defined bond types: {all_bond_types}, detected bond types: {detected_bond_types}

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

"""


def all_bond_lengths():
    """Function to return all bond lengths"""
    bond_lengths_data = {"HC": 1.09, "HO": 0.96, "HN": 1.01}
    bond_lengths = {}
    # Sort each bond_type to ensure uniqueness
    for bond_type in bond_lengths_data:
        bond_type_sorted = "".join(sorted(bond_type))
        bond_lengths[bond_type_sorted] = bond_lengths_data[bond_type]
    return bond_lengths


NUMBERS_TO_SPECIES = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4, 9: 5, 17: 6}
NUMBERS_TO_LMP_TYPES = {1: 1, 6: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7}
BOND_LENGTHS = all_bond_lengths()


def get_bonds_by_type(atoms, bond_types):
    """Function to get bonds by type"""
    if not bond_types:
        return {}, 0

    analysis = Analysis(atoms)

    bonds_by_type = {}

    for bond_type in bond_types:
        bonds = analysis.get_bonds(bond_type[0], bond_type[1])[0]
        bonds_by_type[bond_type] = bonds

    num_bonds = sum([len(bonds) for bonds in bonds_by_type.values()])
    return bonds_by_type, num_bonds


def generate_header(mol, num_atoms, bond_types, bonds_by_type, num_bonds):
    """Function to generate the header for LAMMPS data file"""
    cell = mol.cell
    all_bond_types = ",".join(["-".join(bond_type) for bond_type in bond_types])
    detected_bond_types = ",".join(
        [
            "-".join(bond_type)
            for bond_type in bonds_by_type.keys()
            if bonds_by_type[bond_type]
        ]
    )

    # Determine whether we need to center the cell
    positions = mol.get_positions()
    positions_x_min = positions[:, 0].min()
    xlen_quarter = (cell.lengths() / 4)[0]
    # center is True if it is negative and it is less tha xlen_quarter
    center = True if positions_x_min < (-xlen_quarter) else False

    data = ""
    if center:
        xlen_half, ylen_half, zlen_half = cell.lengths() / 2
        data += HEADER.format(
            num_atoms=num_atoms,
            xlo=-xlen_half,
            xhi=xlen_half,
            ylo=-ylen_half,
            yhi=ylen_half,
            zlo=-zlen_half,
            zhi=zlen_half,
            num_bonds=num_bonds,
            num_bond_types=len(bond_types),
            all_bond_types=all_bond_types,
            detected_bond_types=detected_bond_types,
        )
    else:
        xlen, ylen, zlen = cell.lengths()
        data += HEADER.format(
            num_atoms=num_atoms,
            xlo=0.0,
            xhi=xlen,
            ylo=0.0,
            yhi=ylen,
            zlo=0.0,
            zhi=zlen,
            num_bonds=num_bonds,
            num_bond_types=len(bond_types),
            all_bond_types=all_bond_types,
            detected_bond_types=detected_bond_types,
        )

    # Print cell info
    print(f"Box information:\nlengths: {cell.lengths()}\nangles: {cell.angles()}")
    print(
        f"Assume the box is {'not ' if not center else ''}centered in origin, because x_min is {positions_x_min}\n"
    )
    if not np.array_equal(cell.angles(), [90.0, 90.0, 90.0]):
        warnings.warn(
            "This is not an orthogonal simulation box, please modify the header!"
        )
    print(f"Generated header is the following:\n{data}")

    return data


def generate_atoms_data(mol, num_atoms, bond_types):
    """Function to generate atoms data section for LAMMPS data file"""
    data = "Atoms\n\n"
    positions = mol.get_positions()
    symbols = mol.get_chemical_symbols()
    numbers = mol.get_atomic_numbers()
    residuenumbers = mol.get_array("residuenumbers")
    types = [NUMBERS_TO_LMP_TYPES[i] for i in numbers]
    for i in range(num_atoms):
        position = positions[i]
        if bond_types:
            line = f"{i+1}\t{residuenumbers[i]}\t{types[i]}\t{position[0]}\t{position[1]}\t{position[2]}\t# {symbols[i]}\n"
        else:
            line = f"{i+1}\t{types[i]}\t{position[0]}\t{position[1]}\t{position[2]}\t# {symbols[i]}\n"
        data += line
    return data


def generate_bonds_data(mol, bond_types):
    """Function to generate bonds data section for LAMMPS data file"""
    data = ""
    bonds_by_type, num_bonds = get_bonds_by_type(mol, bond_types=bond_types)

    # add bonds into data
    if bond_types:
        data += "\nBonds\n\n"
        index = 1
        bond_type_index = 1
        for bond_type, bonds in bonds_by_type.items():
            for bond in bonds:
                line = f"{index}\t{bond_type_index}\t{bond[0]+1}\t{bond[1]+1}\t# {bond_type[0]}-{bond_type[1]}\n"
                data += line
                index += 1
            bond_type_index += 1

        data += "\nBond Coeffs\n\n"
        index = 1
        for bond_type, bonds in bonds_by_type.items():
            key = "".join(sorted(bond_type))
            bond_length = BOND_LENGTHS[key]
            line = f"{index}\t{bond_length}\t# {bond_type[0]}-{bond_type[1]}\n"
            data += line
            index += 1

    return bonds_by_type, num_bonds, data


# TODO remove system_size


def generate_data(input_file, output_file, system_size=None, bond_types=[]):

    mol = read(input_file)
    if system_size is None or system_size > len(mol):
        num_atoms = len(mol)
    else:
        num_atoms = system_size

    bonds_by_type, num_bonds, bonds_data = generate_bonds_data(mol, bond_types)
    header_data = generate_header(mol, num_atoms, bond_types, bonds_by_type, num_bonds)
    atoms_data = generate_atoms_data(mol, num_atoms, bond_types)

    data = header_data + atoms_data + bonds_data
    # write out data
    with open(output_file, "w") as file:
        file.write(data)


def convert_and_validate_bond_string(input_string):
    """Function to validate the bond string provided by the user"""
    if len(input_string) == 0:
        return {}

    # Split the input string using the semicolon delimiter
    bonds = input_string.split(",")

    # Split each substring using the empty string delimiter and create pairs of elements
    result = []
    for bond_type in bonds:
        assert len(bond_type) == 2, print(f"wrong format: {input_string}")
        # sort it to make sure it is unique
        sorted_bond_type = "".join(sorted(bond_type))
        assert sorted_bond_type in BOND_LENGTHS, print(
            f"bond type {bond_type} is not supported"
        )
        result.append(bond_type)

    return result


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
        Convert pdb to lammps data format
        Example usage:
        1. only atomic data
            python pdb2lmp.py abc.pdb abc.data
        2. with also bonds data
            python pdb2lmp.py abc.pdb abc.data --bonds OH,CH,NH
        """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("in_file", type=str)
    parser.add_argument("out_file", type=str)
    parser.add_argument("--system_size", type=int, default=None)
    parser.add_argument("--bonds", type=str, default="")
    args = parser.parse_args()

    # Validate bond types
    bond_types = convert_and_validate_bond_string(args.bonds)

    generate_data(args.in_file, args.out_file, args.system_size, bond_types)
