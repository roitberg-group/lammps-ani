import torch
import torchani
import os
import pytest
import lammps_ani
import save_ani

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
use_repulsion_params = [
    pytest.param(False, id="repulsion-no"),
    pytest.param(True, id="repulsion-yes"),
]

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
@pytest.mark.parametrize("use_repulsion", use_repulsion_params)
def test_lmp_with_ase(
        kokkos: bool, use_cuaev: bool, precision: str, nbr: str, pbc: bool, device: str, num_tasks: int, use_repulsion: bool):
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

    ani_model = "ani2x_repulsion.pt" if use_repulsion else "ani2x.pt"
    # prepare configurations
    ani_aev_str = "cuaev" if use_cuaev else "pyaev"
    var_dict = {
        "newton_pair": "off",
        "data_file": "water-0.8nm.data",
        "change_box": "'all boundary p p p'",
        "ani_model_file": ani_model,
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

    # setup ase calculator
    # use cpu for reference result if not for cuaev
    # device = torch.device("cuda") if use_cuaev else torch.device("cpu")
    # ani2x = torchani.models.ANI2x(
    #     periodic_table_index=True,
    #     model_index=None,
    #     cell_list=False,
    #     use_cuaev_interface=use_cuaev,
    #     use_cuda_extension=use_cuaev,
    # )
    ani2x = save_ani.ANI2xRef(use_cuaev=use_cuaev, use_repulsion=use_repulsion)
    # When using half nbrlist, we have to set the cutoff as 7.1 to match lammps nbr cutoff.
    # When using full nbrlist with nocuaev, it is actually still using half_nbr, we also need 7.1 cutoff.
    # Full nbrlist still uses 5.1, which is fine.
    half_nbr = nbr == "half"
    if half_nbr or (not half_nbr and not use_cuaev):
        ani2x.model.aev_computer.neighborlist.cutoff = 7.1
    use_double = precision == "double"
    dtype = torch.float64 if use_double else torch.float32
    # calculator = ani2x.to(dtype).to(device).ase()
    calculator = torchani.ase.Calculator(ani2x.to(dtype).to(device))

    # run ase
    aserunner = lammps_ani.utils.AseRunner("water-0.8nm.pdb", calculator=calculator, pbc=pbc)
    ase_traj = aserunner.run()

    # check result
    lammps_ani.utils.compare_lmp_ase(lmp_dump, ase_traj, high_prec=use_double)
