import torch
import torchani
import os
import ase
import pytest
import yaml
import subprocess
import numpy as np
from typing import Dict
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase import units
from .models import all_models


LAMMPS_PATH = os.path.join(os.environ["LAMMPS_ROOT"], "build/lmp_mpi")
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
        stdout = subprocess.run(self.run_commands, shell=True, check=True, stdout=subprocess.DEVNULL)
        with open(self.var_dict["dump_file"], "r") as stream:
            documents = list(yaml.safe_load_all(stream))
        return documents


class AseRunner():
    def __init__(self, input_file: str, calculator: ase.calculators.calculator.Calculator, pbc: bool = False):
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
    rtol = 1e-5 if high_prec else 1e-3
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
        print("force error: ", np.abs(ase_force - lmp_force).max())
        assert np.allclose(ase_force, lmp_force, rtol, atol)
        assert np.allclose(ase_pos, lmp_pos, rtol, atol)
        if i > 0:
            assert np.allclose(lmp_potEng, ase_potEng, rtol, atol)


pbc_params = [
    pytest.param(True, id="pbc_true"),
    pytest.param(False, id="pbc_false"),
]
precision_params = [
    pytest.param("single", id="precision_single"),
    pytest.param("double", id="precision_double"),
]
num_tasks_params = [
    pytest.param(1, id="num_tasks_1"),
    pytest.param(2, id="num_tasks_2")
]

modelfile_params = all_models.keys()
# remove modelfiles that have unittest as False
modelfile_params = [modelfile for modelfile in modelfile_params if all_models[modelfile]["unittest"]]

@pytest.mark.parametrize("pbc", pbc_params)
@pytest.mark.parametrize("precision", precision_params)
@pytest.mark.parametrize("num_tasks", num_tasks_params)
@pytest.mark.parametrize(
    "kokkos, use_cuaev, nbr, device",
    [
        # kokkos on, only support full nbr
        # kokkos works with cuaev (only cuda), nocuaev (cuda and cpu)
        pytest.param(
            True, True, "full", "cuda", id="kokkos_full_cuaev"
        ),
        pytest.param(
            True, False, "full", "cuda", id="kokkos_full_nocuaev_cuda"
        ),
        pytest.param(
            True, False, "full", "cpu", id="kokkos_full_nocuaev_cpu"
        ),
        # kokkos off, cuaev, works with full and half nbr, only support cuda
        pytest.param(
            False, True, "full", "cuda", id="cuaev_full_cuda"
        ),
        pytest.param(
            False, True, "half", "cuda", id="cuaev_half_cuda"
        ),
        # kokkos off, nocuaev, works with full and half nbr
        pytest.param(
            False, False, "half", "cuda", id="nocuaev_half_cuda"
        ),
        pytest.param(
            False, False, "full", "cuda", id="nocuaev_full_cuda"
        ),
        pytest.param(
            False, False, "half", "cpu", id="nocuaev_half_cpu"
        ),
        # full nbr on cpu actually just manually convert full to half nbr
        pytest.param(
            False, False, "full", "cpu", id="nocuaev_full_cpu"
        ),
    ],
)
@pytest.mark.parametrize("modelfile", modelfile_params)
def test_lmp_with_ase(
        kokkos: bool, use_cuaev: bool, precision: str, nbr: str, pbc: bool, device: str, num_tasks: int,
        modelfile: str):
    # SKIP: compiled kokkos only work on Ampere GPUs
    SM = torch.cuda.get_device_capability(0)
    SM = int(f'{SM[0]}{SM[1]}')
    if kokkos and SM < 80:
        pytest.skip("compiled kokkos only work on Ampere GPUs")

    # SKIP
    run_github_action_multi = "TEST_WITH_MULTI_PROCS" in os.environ and os.environ["TEST_WITH_MULTI_PROCS"] == "true"
    run_slurm_multi = "SLURM_NTASKS" in os.environ and int(os.environ["SLURM_NTASKS"]) > 1
    if num_tasks > 1 and (not run_github_action_multi) and (not run_slurm_multi):
        pytest.skip("Skip running on 2 MPI Processes")

    # prepare configurations
    ani_aev_str = "cuaev" if use_cuaev else "pyaev"
    var_dict = {
        "newton_pair": "off",
        "data_file": "water-0.8nm.data",
        "change_box": "'all boundary p p p'",
        "ani_model_file": modelfile,
        "ani_device": device,
        "ani_num_models": -1,
        "ani_aev": ani_aev_str,
        "ani_neighbor": nbr,
        "ani_precision": precision
    }
    if not pbc:
        var_dict["change_box"] = "'all boundary f f f'"
    if kokkos:
        var_dict["newton_pair"] = "on"

    # run lammps
    lmprunner = LammpsRunner(LAMMPS_PATH, "in.lammps", var_dict, kokkos, num_tasks)
    lmp_dump = lmprunner.run()

    def set_ref_cuda_aev(model, use_cuaev):
        model.aev_computer.use_cuaev_interface = use_cuaev
        model.aev_computer.use_cuda_extension = use_cuaev
        return model

    # setup ase calculator
    # use_repulsion = all_models[modelfile]["use_repulsion"]
    ref_model = all_models[modelfile]["model"]()
    ref_model = set_ref_cuda_aev(ref_model, use_cuaev)
    # When using half nbrlist, we have to set the cutoff as 7.1 to match lammps nbr cutoff.
    # When using full nbrlist with nocuaev, it is actually still using half_nbr, we also need 7.1 cutoff.
    # Full nbrlist still uses 5.1, which is fine.
    half_nbr = nbr == "half"
    if half_nbr or (not half_nbr and not use_cuaev):
        ref_model.aev_computer.neighborlist.cutoff = 7.1
    use_double = precision == "double"
    dtype = torch.float64 if use_double else torch.float32
    # calculator = ref_model.to(dtype).to(device).ase()
    calculator = torchani.ase.Calculator(ref_model.to(dtype).to(device))

    # run ase
    aserunner = AseRunner("water-0.8nm.pdb", calculator=calculator, pbc=pbc)
    ase_traj = aserunner.run()

    # check result
    compare_lmp_ase(lmp_dump, ase_traj, high_prec=use_double)
