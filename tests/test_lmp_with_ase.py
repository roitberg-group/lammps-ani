import torch
import torchani
import os
import pytest
import lammps_ani


LAMMPS_PATH = os.path.join(os.environ["LAMMPS_ROOT"], "build/lmp_mpi")

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


@pytest.mark.parametrize("pbc", pbc_params)
@pytest.mark.parametrize("precision", precision_params)
@pytest.mark.parametrize("num_tasks", num_tasks_params)
@pytest.mark.parametrize(
    "kokkos, cuaev, nbr, device",
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
def test_lmp_with_ase(
        kokkos: bool, cuaev: bool, precision: str, nbr: str, pbc: bool, device: str, num_tasks: int):
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
    ani_aev_str = "cuaev" if cuaev else "pyaev"
    var_dict = {
        "newton_pair": "off",
        "data_file": "water-0.8nm.data",
        "change_box": "'all boundary p p p'",
        "ani_model_file": "ani2x.pt",
        "ani_device": device,
        "ani_num_models": 8,
        "ani_aev": ani_aev_str,
        "ani_neighbor": nbr,
        "ani_precision": precision
    }
    if not pbc:
        var_dict["change_box"] = "'all boundary f f f'"
    if kokkos:
        var_dict["newton_pair"] = "on"

    # run lammps
    lmprunner = lammps_ani.utils.LammpsRunner(LAMMPS_PATH, "in.lammps", var_dict, kokkos, num_tasks)
    lmp_dump = lmprunner.run()

    # run ase
    use_double = precision == "double"
    half_nbr = nbr == "half"
    aserunner = lammps_ani.utils.AseRunner("water-0.8nm.pdb", pbc=pbc, use_double=use_double, use_cuaev=cuaev, half_nbr=half_nbr)
    ase_traj = aserunner.run()

    # check result
    high_prec = not cuaev and use_double
    lammps_ani.utils.compare_lmp_ase(lmp_dump, ase_traj, high_prec)
