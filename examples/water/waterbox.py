import argparse
from openmm.app import *
from openmm import *
from openmm.unit import *
from simtk import unit
from openmmtools import testsystems

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("box_size", type=float, help="water box size in nanometer")
    args = parser.parse_args()

    box_size = args.box_size
    water_box = testsystems.WaterBox(box_edge=box_size * unit.nanometer)
    positions = water_box.positions
    atoms = list(water_box.topology.atoms())
    elements = [a.element.symbol for a in atoms]
    pdbfile = f"water-{box_size}nm.pdb"
    PDBFile.writeFile(water_box.topology, positions, open(pdbfile, "w"))
    print(f"check pdbfile: {pdbfile}")
