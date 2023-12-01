from ase.io import read, write

mol = ase.io.read("mixture_22800000.pdb")
ase.io.write("mixture_22800000_ase.pdb", mol)
