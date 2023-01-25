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
    def __init__(self, input_file: str, calculator, pbc: bool = False):
        atoms = read(input_file)

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

