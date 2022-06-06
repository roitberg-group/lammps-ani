import ase
import torch
import torchani
import numpy as np
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase import units
import argparse

torch.set_printoptions(precision=15)
np.set_printoptions(precision=15)


def run(pbc=False):
    input_file = "water-0.8nm.pdb"
    atoms = read(input_file)

    device = torch.device("cuda")
    calculator = (
        torchani.models.ANI2x(
            periodic_table_index=True,
            model_index=None,
            cell_list=True,
            use_cuaev_interface=True,
            use_cuda_extension=True,
        )
        .to(device)
        .ase()
    )

    print(len(atoms), "atoms in the cell")
    atoms.set_calculator(calculator)
    if not pbc:
        atoms.set_pbc([False, False, False])

    hartree2kcalmol = 627.5094738898777
    atoms.get_kinetic_energy

    def printenergy(a=atoms):
        """Function to print the potential, kinetic and total energy."""
        epot = a.get_potential_energy() / units.Hartree * hartree2kcalmol
        ekin = a.get_kinetic_energy() / units.Hartree * hartree2kcalmol
        forces = atoms.get_forces().astype(np.double) / units.Hartree * hartree2kcalmol
        print(
            "Energy: Epot = %.13f kcal/mol  Ekin = %.13f kcal/mol (T=%3.0fK)  "
            "Etot = %.13f kcal/mol"
            % (epot, ekin, ekin / len(a) / (1.5 * units.kB), epot + ekin)
        )
        print(f"forces: \n{forces}")

    dyn = VelocityVerlet(
        atoms, dt=0.1 * units.fs, trajectory="md.traj", logfile="md.log"
    )
    dyn.attach(printenergy, interval=1)
    print("Beginning dynamics...")
    dyn.run(4)  # take 1000 steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pbc', default=False, action='store_true')
    args = parser.parse_args()

    run(args.pbc)
