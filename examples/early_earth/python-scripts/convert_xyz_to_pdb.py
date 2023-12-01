import argparse
from ase.io import read, write

# Create the parser
parser = argparse.ArgumentParser(description='Convert an XYZ file to a PDB file.')

# Add the arguments
parser.add_argument('input_file', type=str, help='The input XYZ file.')
parser.add_argument('output_file', type=str, help='The output PDB file.')

# Parse the arguments
args = parser.parse_args()

# Read the atoms from the input file
atoms = read(args.input_file)

# Write the atoms to the output file
write(args.output_file, atoms)

