import re
import os
import subprocess
from typing import Dict

import yaml
import torch
import pytest
import numpy as np
import pandas as pd
import ase
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase import units
from torchani.ase import Calculator as ANICalculator

from .models import all_models

np.set_printoptions(precision=12)
LAMMPS_PATH = os.path.join(os.environ["LAMMPS_ROOT"], "build/lmp_mpi")
STEPS = 4


class LammpsRunner:
    def __init__(
        self,
        lmp: str,
        input_file: str,
        var_dict: Dict,
        kokkos: bool,
        num_tasks: int = 1,
    ):
        var_dict["dump_file"] = "dump.yaml"
        var_dict["logfile"] = "log.lammps"
        var_commands = " ".join(
            [f"-var {var} {value}" for var, value in var_dict.items()]
        )
        kokkos_commands = f"-k on g {num_tasks} -sf kk" if kokkos else ""
        run_commands = f"mpirun -np {num_tasks} {lmp} {var_commands} -var steps {STEPS} {kokkos_commands} -in {input_file} -log {var_dict['logfile']}"
        print(f"\n{run_commands}")
        self.run_commands = run_commands
        self.var_dict = var_dict

    def run(self):
        subprocess.run(
            self.run_commands, shell=True, check=True, stdout=subprocess.DEVNULL
        )
        df_thermo = self.read_thermo_from_log(self.var_dict["logfile"])

        with open(self.var_dict["dump_file"], "r") as stream:
            dump_data = list(yaml.safe_load_all(stream))

        return dump_data, df_thermo

    @staticmethod
    def read_thermo_from_log(log_file):
        docs = ""
        # there is an embeded thermo yaml in the log file, we need to extract it
        with open(log_file) as f:
            for line in f:
                m = re.search(r"^(keywords:.*$|data:$|---$|\.\.\.$|  - \[.*\]$)", line)
                if m:
                    docs += m.group(0) + '\n'
        thermo = list(yaml.safe_load_all(docs))
        df_thermo = pd.DataFrame(data=thermo[0]['data'], columns=thermo[0]['keywords'])
        return df_thermo


class AseRunner:
    def __init__(
        self,
        input_file: str,
        calculator: ase.calculators.calculator.Calculator,
        pbc: bool = False,
    ):
        atoms = read(input_file)

        print(len(atoms), "atoms in the cell")
        atoms.calc = calculator
        if not pbc:
            atoms.set_pbc([False, False, False])

        hartree2kcalmol = 627.5094738898777

        def printenergy(a=atoms):
            """Function to print the potential, kinetic and total energy."""
            epot = a.get_potential_energy() / units.Hartree * hartree2kcalmol
            ekin = a.get_kinetic_energy() / units.Hartree * hartree2kcalmol
            # forces = atoms.get_forces().astype(np.double) / units.Hartree * hartree2kcalmol
            print(
                "Energy: Epot = %.13f kcal/mol  Ekin = %.13f kcal/mol (T=%3.4fK)  "
                "Etot = %.13f kcal/mol" % (epot, ekin, a.get_temperature(), epot + ekin)
            )

        # create a traj, so we could dump stress data
        self.traj = Trajectory('md.traj', 'w', atoms, properties=["energy", "forces", "stress"])
        dyn = VelocityVerlet(atoms, timestep=0.1 * units.fs)
        dyn.attach(printenergy, interval=1)
        dyn.attach(self.traj.write, interval=1)

        self.dyn = dyn

    def run(self):
        print("Beginning dynamics...")
        self.dyn.run(STEPS)

        # close traj and read it back
        self.traj.close()
        traj = list(Trajectory("md.traj"))
        return traj


def compare_lmp_ase(lmp_data, ase_traj, high_prec, using_cuaev):
    lmp_dump, lmp_df_thermo = lmp_data
    atol = 1e-9 if high_prec else 1e-3
    rtol = 1e-5 if high_prec else 1e-3
    num_traj = len(ase_traj)
    for i in range(num_traj):
        lmp_data = lmp_dump[i]
        lmp_potEng = lmp_df_thermo["PotEng"].iloc[i]
        lmp_volume = lmp_df_thermo["Volume"].iloc[i]
        lmp_temp = lmp_df_thermo["Temp"].iloc[i]
        lmp_index = np.array(lmp_data["data"])[:, 0]
        lmp_data["data"] = np.array(lmp_data["data"])[np.argsort(lmp_index)]
        lmp_pos = np.array(lmp_data["data"])[:, 2:5]
        lmp_force = np.array(lmp_data["data"])[:, 5:]
        # To compare stress, we convert LAMMPS and ASE stress both into unit of kcal/mol/A^3
        # We compare stresses with kinetic part included.
        # Note that ASE voigt formart is (xx, yy, zz, yz, xz, xy), whereas in LAMMPS it is (xx, yy, zz, xy, xz, yz)
        lmp_press = lmp_df_thermo.loc[lmp_df_thermo['Step'] == i, ['c_press[1]', 'c_press[2]', 'c_press[3]', 'c_press[6]', 'c_press[5]', 'c_press[4]']].values.tolist()
        lmp_press = np.array(lmp_press)
        nktv2p = 68568.415
        lmp_stress = lmp_press / nktv2p * lmp_volume

        ase_atoms = ase_traj[i]
        hartree2kcalmol = 627.5094738898777
        ase_temp = ase_atoms.get_temperature()
        ase_pos = ase_atoms.positions
        ase_force = ase_atoms.get_forces() / units.Hartree * hartree2kcalmol
        ase_potEng = ase_atoms.get_potential_energy() / units.Hartree * hartree2kcalmol
        ase_stress = ase_atoms.get_stress(include_ideal_gas=True) / units.Hartree * hartree2kcalmol * ase_atoms.get_volume()
        ase_stress = - ase_stress

        print("force error: ", np.abs(ase_force - lmp_force).max())
        # compare force
        assert np.allclose(ase_force, lmp_force, rtol, atol)
        # compare position
        assert np.allclose(ase_pos, lmp_pos, rtol, atol)
        # compare temperature
        assert np.allclose(lmp_temp, ase_temp, atol=1.3e-1)
        # compare potential energy
        assert np.allclose(lmp_potEng, ase_potEng, rtol, atol)

        # compare stress for pyaev, cuaev currently does not support stress
        if not using_cuaev:
            assert np.allclose(ase_stress, lmp_stress, rtol, atol)


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
    pytest.param(2, id="num_tasks_2"),
]

# remove modelfiles that have unittest as False
modelfile_params = [k for k, value in all_models.items() if value["unittest"]]


@pytest.mark.parametrize("pbc", pbc_params)
@pytest.mark.parametrize("precision", precision_params)
@pytest.mark.parametrize("num_tasks", num_tasks_params)
@pytest.mark.parametrize(
    "kokkos, use_cuaev, nbr, device",
    [
        # kokkos on, only support full nbr
        # kokkos works with cuaev (only cuda), nocuaev (cuda and cpu)
        pytest.param(True, True, "full", "cuda", id="kokkos_full_cuaev"),
        pytest.param(True, False, "full", "cuda", id="kokkos_full_nocuaev_cuda"),
        pytest.param(True, False, "full", "cpu", id="kokkos_full_nocuaev_cpu"),
        # kokkos off, cuaev, works with full and half nbr, only support cuda
        pytest.param(False, True, "full", "cuda", id="cuaev_full_cuda"),
        pytest.param(False, True, "half", "cuda", id="cuaev_half_cuda"),
        # kokkos off, nocuaev, works with full and half nbr
        pytest.param(False, False, "half", "cuda", id="nocuaev_half_cuda"),
        pytest.param(False, False, "full", "cuda", id="nocuaev_full_cuda"),
        pytest.param(False, False, "half", "cpu", id="nocuaev_half_cpu"),
        # full nbr on cpu actually just manually convert full to half nbr
        pytest.param(False, False, "full", "cpu", id="nocuaev_full_cpu"),
    ],
)
@pytest.mark.parametrize("modelfile", modelfile_params)
def test_lmp_with_ase(
    kokkos: bool,
    use_cuaev: bool,
    precision: str,
    nbr: str,
    pbc: bool,
    device: str,
    num_tasks: int,
    modelfile: str,
):
    # SKIP: compiled kokkos only work on Ampere GPUs
    _SM = torch.cuda.get_device_capability(0)
    SM = int(f"{_SM[0]}{_SM[1]}")
    if kokkos and SM < 80:
        pytest.skip("compiled kokkos only work on Ampere GPUs")

    # SKIP
    run_github_action_multi = (
        "TEST_WITH_MULTI_PROCS" in os.environ
        and os.environ["TEST_WITH_MULTI_PROCS"] == "true"
    )
    run_slurm_multi = (
        "SLURM_NTASKS" in os.environ and int(os.environ["SLURM_NTASKS"]) > 1
    )
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
        "ani_precision": precision,
    }
    if not pbc:
        var_dict["change_box"] = "'all boundary f f f'"
    if kokkos:
        var_dict["newton_pair"] = "on"

    # run lammps
    lmprunner = LammpsRunner(LAMMPS_PATH, "in.lammps", var_dict, kokkos, num_tasks)
    lmp_data = lmprunner.run()

    def set_ref_cuda_aev(model, use_cuaev):
        model.aev_computer.use_cuaev_interface = use_cuaev
        model.aev_computer.use_cuda_extension = use_cuaev
        return model

    # setup ase calculator
    model_info = all_models[modelfile]
    if "kwargs" in model_info:
        kwargs = model_info["kwargs"]
    else:
        kwargs = {}
    ref_model = model_info["model"](**kwargs)

    ref_model = set_ref_cuda_aev(ref_model, use_cuaev)
    # When using half nbrlist, we have to set the cutoff as 7.1 to match lammps nbr cutoff.
    # When using full nbrlist with nocuaev, it is actually still using half_nbr, we also need 7.1 cutoff.
    # Full nbrlist still uses 5.1, which is fine.
    half_nbr = nbr == "half"
    if half_nbr or (not half_nbr and not use_cuaev):
        ref_model.aev_computer.neighborlist.cutoff = 7.1
    use_double = precision == "double"
    dtype = torch.float64 if use_double else torch.float32
    calculator = ANICalculator(ref_model.to(dtype).to(device))

    # run ase
    aserunner = AseRunner("water-0.8nm.pdb", calculator=calculator, pbc=pbc)
    ase_traj = aserunner.run()

    # check result
    compare_lmp_ase(lmp_data, ase_traj, high_prec=use_double, using_cuaev=use_cuaev)
