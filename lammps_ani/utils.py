import torch
import torchani
import yaml
from typing import Dict
import subprocess
import numpy as np
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase import units

STEPS = 4

class LammpsRunner():
    def __init__(self, lmp: str, input_file: str, var_dict: Dict, kokkos: bool, num_tasks: int = 1):
        var_dict["dump_file"] = "dump.yaml"
        var_commands = " ".join([f"-var {var} {value}" for var, value in var_dict.items()])
        kokkos_commands = f"-k on g {num_tasks} -sf kk" if kokkos else ""
        run_commands = f"mpirun -np {num_tasks} {lmp} {var_commands} -var steps {STEPS} {kokkos_commands} -in {input_file}"
        print(f"\n{run_commands}")
        self.run_commands = run_commands
        self.var_dict = var_dict

    def run(self):
        stdout = subprocess.run(self.run_commands, shell=True, check=True)
        with open(self.var_dict["dump_file"], "r") as stream:
            documents = list(yaml.safe_load_all(stream))
        return documents


class AseRunner():
    def __init__(self, input_file: str, pbc: bool = False, use_double: bool = True, use_cuaev: bool = False, half_nbr: bool = True):
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
        # When using half nbrlist, we have to set the cutoff as 7.1 to match lammps nbr cutoff.
        # When using full nbrlist with nocuaev, it is actually still using half_nbr, we also need 7.1 cutoff.
        # Full nbrlist still uses 5.1, which is fine.
        if half_nbr or (not half_nbr and not use_cuaev):
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
                "Energy: Epot = %.13f kcal/mol  Ekin = %.13f kcal/mol (T=%3.2fK)  "
                "Etot = %.13f kcal/mol"
                % (epot, ekin, a.get_temperature(), epot + ekin)
            )

        dyn = VelocityVerlet(
            atoms, timestep=0.1 * units.fs, trajectory="md.traj", logfile="md.log"
        )
        dyn.attach(printenergy, interval=1)
        self.dyn = dyn

    def run(self):
        print("Beginning dynamics...")
        self.dyn.run(STEPS)  # take 1000 steps
        traj = list(Trajectory('md.traj'))
        return traj


def compare_lmp_ase(lmp_dump, ase_traj, high_prec):
    # with open("tests/dump.yaml", "r") as stream:
    #     lmp_dump = list(yaml.safe_load_all(stream))
    # ase_traj = list(Trajectory('tests/md.traj'))
    atol = 1e-9 if high_prec else 1e-3
    rtol = 1e-6 if high_prec else 1e-3
    num_traj = len(ase_traj)
    for i in range(num_traj):
        lmp_data = lmp_dump[i]
        lmp_potEng = lmp_data["thermo"][1]["data"][1]
        lmp_index = np.array(lmp_data["data"])[:, 0]
        lmp_data["data"] = np.array(lmp_data["data"])[np.argsort(lmp_index)]
        lmp_pos = np.array(lmp_data["data"])[:, 2:5]
        lmp_force = np.array(lmp_data["data"])[:, 5:]

        ase_atoms = ase_traj[i]
        hartree2kcalmol = 627.5094738898777
        ase_pos = ase_atoms.positions
        ase_force = ase_atoms.get_forces() / units.Hartree * hartree2kcalmol
        ase_potEng = ase_atoms.get_potential_energy() / units.Hartree * hartree2kcalmol
        print(np.abs(ase_force - lmp_force).max())
        assert np.allclose(ase_force, lmp_force, rtol, atol)
        assert np.allclose(ase_pos, lmp_pos, rtol, atol)
        if i > 0:
            assert np.allclose(lmp_potEng, ase_potEng, rtol, atol)

