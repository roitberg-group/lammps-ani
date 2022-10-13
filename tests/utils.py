import torch
import torchani
import os
import tempfile
import yaml
from typing import Dict
import subprocess
import numpy as np
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase import units

import torchsnooper
import snoop
torchsnooper.register_snoop(verbose=True)


class LammpsRunner():
    # @snoop
    def __init__(self, lmp: str, input_file: str, var_dict: Dict):
        var_dict["dump_file"] = "dump.yaml"
        var_commands = " ".join([f"-var {var} {value}" for var, value in var_dict.items()])
        run_commands = f"mpirun -np 1 {lmp} {var_commands} -in {input_file}"
        print(run_commands)
        self.run_commands = run_commands
        self.var_dict = var_dict

    def run(self):
        stdout = subprocess.run(self.run_commands, shell=True, stdout=subprocess.PIPE, check=True)
        with open(self.var_dict["dump_file"], "r") as stream:
            documents = list(yaml.safe_load_all(stream))
        return documents


class AseRunner():
    def __init__(self, pbc=False, use_double=True, use_cuaev=False):
        input_file = "water-0.8nm.pdb"
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

        def printenergy(a=atoms):
            """Function to print the potential, kinetic and total energy."""
            epot = a.get_potential_energy() / units.Hartree * hartree2kcalmol
            ekin = a.get_kinetic_energy() / units.Hartree * hartree2kcalmol
            # forces = atoms.get_forces().astype(np.double) / units.Hartree * hartree2kcalmol
            print(
                "Energy: Epot = %.13f kcal/mol  Ekin = %.13f kcal/mol (T=%3.0fK)  "
                "Etot = %.13f kcal/mol"
                % (epot, ekin, ekin / len(a) / (1.5 * units.kB), epot + ekin)
            )

        dyn = VelocityVerlet(
            atoms, timestep=0.1 * units.fs, trajectory="md.traj", logfile="md.log"
        )
        dyn.attach(printenergy, interval=1)
        self.dyn = dyn

    def run(self):
        print("Beginning dynamics...")
        self.dyn.run(4)  # take 1000 steps
        traj = list(Trajectory('md.traj'))
        return traj


def compare_lmp_ase(lmp_dump, ase_traj):
    # with open("tests/dump.yaml", "r") as stream:
    #     lmp_dump = list(yaml.safe_load_all(stream))
    # ase_traj = list(Trajectory('tests/md.traj'))
    num_traj = len(ase_traj)
    for i in range(num_traj):
        lmp_data = lmp_dump[i]
        ase_atoms = ase_traj[i]
        lmp_potEng = lmp_data["thermo"][1]["data"][8]
        lmp_pos = np.array(lmp_data["data"])[:, 2:5]
        lmp_force = np.array(lmp_data["data"])[:, 5:]
        hartree2kcalmol = 627.5094738898777
        ase_pos = ase_atoms.positions
        ase_force = ase_atoms.get_forces() / units.Hartree * hartree2kcalmol
        ase_potEng = ase_atoms.get_potential_energy()/ units.Hartree * hartree2kcalmol

lmp = os.path.join(os.environ["LAMMPS_ROOT"], "build/lmp_mpi")
var_dict = {
    "newton_pair": "off",
    "num_models": 8,
    "data_file": "water-0.8nm.data",
    "model_file": "ani2x_cuaev_single_full.pt",
    "device": "cuda"
}
lmprunner = LammpsRunner(lmp, "in.lammps", var_dict)
lmp_dump = lmprunner.run()
print(lmp_dump)

aserunner = AseRunner(pbc=True, use_double=False, use_cuaev=True)
ase_traj = aserunner.run()
print(ase_traj)
