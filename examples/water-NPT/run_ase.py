import os
import ase
import torch
import datetime
import torchani
import numpy as np
import pandas as pd
from ase.md import MDLogger
from ase.io import read
from ase.optimize import BFGS
from ase.md.verlet import VelocityVerlet
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
import argparse

torch.set_printoptions(precision=15)
np.set_printoptions(precision=15)
pd.set_option("display.precision", 15)


# TODO how can we just import the model instead of copy-pasting the code?
def ANI2x_Repulsion_Model():
    elements = ("H", "C", "N", "O", "S", "F", "Cl")

    def dispersion_atomics(atom: str = "H"):
        dims_for_atoms = {
            "H": (1008, 256, 192, 160),
            "C": (1008, 256, 192, 160),
            "N": (1008, 192, 160, 128),
            "O": (1008, 192, 160, 128),
            "S": (1008, 160, 128, 96),
            "F": (1008, 160, 128, 96),
            "Cl": (1008, 160, 128, 96),
        }
        return torchani.atomics.standard(
            dims_for_atoms[atom], activation=torch.nn.GELU(), bias=False
        )

    model = torchani.models.ANI2x(
        pretrained=False,
        cutoff_fn="smooth",
        atomic_maker=dispersion_atomics,
        ensemble_size=7,
        repulsion=True,
        repulsion_kwargs={
            "symbols": elements,
            "cutoff": 5.1,
            "cutoff_fn": torchani.aev.cutoffs.CutoffSmooth(order=2),
        },
        periodic_table_index=True,
        model_index=None,
        cell_list=False,
        # use_cuaev_interface=False,
        # use_cuda_extension=False,
    )
    state_dict = torchani.models._fetch_state_dict(
        "anid_state_dict_mod.pt", private=True
    )
    for key in state_dict.copy().keys():
        if key.startswith("potentials.0"):
            state_dict.pop(key)
    for key in state_dict.copy().keys():
        if key.startswith("potentials.1"):
            new_key = key.replace("potentials.1", "potentials.0")
            state_dict[new_key] = state_dict[key]
            state_dict.pop(key)
    for key in state_dict.copy().keys():
        if key.startswith("potentials.2"):
            new_key = key.replace("potentials.2", "potentials.1")
            state_dict[new_key] = state_dict[key]
            state_dict.pop(key)

    model.load_state_dict(state_dict)
    # setup repulsion calculator
    model.rep_calc = model.potentials[0]

    return model


class CustomLogger(MDLogger):
    def __init__(self, dyn, atoms, logfile, stress=False, mode="a"):
        super().__init__(dyn, atoms, logfile, header=False, stress=stress, mode=mode)

        if self.dyn is not None:
            self.hdr = "%-9s " % ("Time[ps]",)
            self.fmt = "%-9.4f "
        else:
            self.hdr = ""
            self.fmt = ""

        self.hdr += "%15s %15s %15s %8s %8s %15s %10s" % (
            "Etot[kcal/mol]",
            "Epot[kcal/mol]",
            "Ekin[kcal/mol]",
            "T[K]",
            "V[A^3]",
            "Density[g/cm^3]",
            "P[atm]",
        )
        self.fmt += "%15.4f %15.4f %15.4f %8.1f %8.1f %15.3f %10.3f"
        if self.stress:
            self.hdr += (
                "      ---------------------- stress [GPa] " "-----------------------"
            )
            self.fmt += 6 * " %10.3f"

        self.fmt += "\n"
        self.logfile.write(self.hdr + "\n")
        self.logfile.flush()

    def __call__(self):
        hartree2kcalmol = 627.5094738898777

        epot = self.atoms.get_potential_energy() / units.Hartree * hartree2kcalmol
        ekin = self.atoms.get_kinetic_energy() / units.Hartree * hartree2kcalmol
        temp = self.atoms.get_temperature()
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000 * units.fs)
            dat = [t]
        else:
            dat = []
        dat += [epot + ekin, epot, ekin, temp]

        # volume and density
        mass = self.atoms.get_masses().sum() / 6.0221408e23
        volume = self.atoms.get_volume()
        density = mass / (volume * 1e-24)

        stress = self.atoms.get_stress(voigt=False, include_ideal_gas=True)
        pressure_Pa = -stress.trace() / 3 / units.Pascal
        pressure_atm = pressure_Pa * 9.86923267e-6

        dat += [volume, density, pressure_atm]

        if self.stress:
            dat += list(self.atoms.get_stress(include_ideal_gas=True) / units.GPa)

        self.logfile.write(self.fmt % tuple(dat))
        self.logfile.flush()


def run(pdb_file, pbc=False, use_double=True, use_cuaev=False, repulsion=False, name=""):
    atoms = read(pdb_file)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if repulsion:
        ani2x = ANI2x_Repulsion_Model()
    else:
        ani2x = torchani.models.ANI2x(
            periodic_table_index=True,
            model_index=None,
            use_cuda_extension=use_cuaev,
        )

    # double precision
    if use_double:
        ani2x = ani2x.double()
    calculator = ani2x.to(device).ase()

    print(len(atoms), "atoms in the cell")
    atoms.set_calculator(calculator)

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    if not pbc:
        atoms.set_pbc([False, False, False])

    # generate filenames
    # Create the directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    now = datetime.datetime.now()
    logfile = 'logs/ase-' + now.strftime("%Y-%m-%d-%H%M%S") + f"-NPT_{pdb_file}-{name}.log"
    trajfile = 'logs/ase-' + now.strftime("%Y-%m-%d-%H%M%S") + f"-NPT_{pdb_file}.trajectories"

    print("Begin minimizing...")
    opt = BFGS(atoms)
    opt.run(fmax=0.05)

    print("Start NPT simulation...")
    dyn = NPTBerendsen(
        atoms,
        timestep=0.1 * units.fs,
        temperature_K=300,
        taut=0.1 * 1000 * units.fs,
        pressure_au=1.0 * units.bar,
        taup=1.0 * 1000 * units.fs,
        compressibility_au=4.57e-5 / units.bar,
        trajectory=trajfile,
        loginterval=100,
    )

    # Create an MDLogger instance to log the properties
    logger = CustomLogger(dyn, atoms, logfile, stress=False)

    dyn.attach(logger, interval=100)
    print("Beginning NPT dynamics...")
    dyn.run(500000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb", type=str)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument('--rep', action='store_true')
    args = parser.parse_args()

    run(args.pdb, True, False, False, args.rep, args.name)
