import ase
import ase.io
from ase import Atoms
import numpy as np

# Load your atoms from the .xyz file
atoms = ase.io.read('start.xyz')

# Get all the atom positions
positions = atoms.get_positions()

# Find the minimum and maximum coordinates in each dimension
min_coords = positions.min(axis=0)
max_coords = positions.max(axis=0)

# The box size is the difference between the maximum and minimum coordinates
box_size = max_coords - min_coords

# Set the cell (box) size
atoms.set_cell([box_size[0], box_size[1], box_size[2]])

# Set the periodic boundary conditions
atoms.set_pbc([True, True, True])

print("Estimated box size: ", box_size)

