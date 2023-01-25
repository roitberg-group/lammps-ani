import ase
import torch
import torchani
import numpy as np
import pandas as pd
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase import units
import argparse

torch.set_printoptions(precision=15)
np.set_printoptions(precision=15)
pd.set_option("display.precision", 15)


def run(pbc=False, use_double=True, use_cuaev=False):
    input_file = "../water-0.8nm.pdb"
    atoms = read(input_file)

    # use cpu for reference result if not for cuaev
    device = torch.device("cuda") if use_cuaev else torch.device("cpu")
    ani2x = torchani.models.ANI2x(
        periodic_table_index=True,
        model_index=None,
        cell_list=False,
        use_cuaev_interface=use_cuaev,
        use_cuda_extension=use_cuaev,
    )
    # TODO It is IMPORTANT to set cutoff as 7.1 to match lammps nbr cutoff
    ani2x.aev_computer.neighborlist.cutoff = 7.1
    # double precision
    if use_double:
        ani2x = ani2x.double()
    calculator = ani2x.to(device).ase()

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
        df = pd.DataFrame(forces)
        df.index = df.index + 1
        print(f"forces: \n{df}")

    dyn = VelocityVerlet(
        atoms, dt=0.1 * units.fs, trajectory="md.traj", logfile="md.log"
    )
    dyn.attach(printenergy, interval=1)
    print("Beginning dynamics...")
    dyn.run(4)  # take 1000 steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbc", default=False, action="store_true")
    parser.add_argument("--single", default=False, action="store_true")
    parser.add_argument("--cuaev", default=False, action="store_true")
    args = parser.parse_args()
    if args.cuaev:
        assert (
            args.single
        ), "please only use single precision (--single) for cuaev reference result"

    run(args.pbc, not args.single, args.cuaev)
