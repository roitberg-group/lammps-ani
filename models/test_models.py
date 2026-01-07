import copy
from pathlib import Path

import ase
from ase.io import read
import pytest
import torch
from torchani.units import hartree2kcalpermol
from torchani.nn import Ensemble

from .ani_models import all_models, save_models

hartree2kcalmol = hartree2kcalpermol(1)

# Get the directory where this module is located
_MODULE_DIR = Path(__file__).parent.absolute()
_TESTS_DIR = _MODULE_DIR.parent / "tests"

# pytest parameter and id
runpbc_params = [
    pytest.param(True, id="pbc_true"),
    pytest.param(False, id="pbc_false"),
]
device_params = [
    pytest.param("cuda", id="cuda"),
    pytest.param("cpu", id="cpu"),
]
use_double_params = [
    pytest.param(True, id="double"),
    pytest.param(False, id="single"),
]
use_cuaev_params = [
    pytest.param(True, id="cuaev"),
    pytest.param(False, id="nocuaev"),
]
use_fullnbr_params = [
    pytest.param(True, id="full"),
    pytest.param(False, id="half"),
]
virial_flag_params = [
    pytest.param(True, id="virial_true"),
    pytest.param(False, id="virial_false"),
]
# remove modelfiles that have unittest as False
modelfile_params = [k for k, value in all_models.items() if value["unittest"]]


# Save all models by using session-scoped "autouse" fixture, this will run ahead of all tests.
@pytest.fixture(scope="session", autouse=True)
def session_start():
    print("Pytest session started, saving all models")
    save_models()


# test all models
@pytest.mark.parametrize("runpbc", runpbc_params)
@pytest.mark.parametrize("device", device_params)
@pytest.mark.parametrize("use_double", use_double_params)
@pytest.mark.parametrize("use_cuaev", use_cuaev_params)
@pytest.mark.parametrize("use_fullnbr", use_fullnbr_params)
@pytest.mark.parametrize("modelfile", modelfile_params)
@pytest.mark.parametrize("virial_flag", virial_flag_params)
def test_models(
    runpbc, device, use_double, use_cuaev, use_fullnbr, modelfile, virial_flag
):
    # when pbc is on, full nbrlist converted from half nbrlist is not correct
    if use_fullnbr and runpbc:
        pytest.skip("Does not support full neighbor list using pyaev when pbc is on")
    if use_cuaev and device == "cpu":
        pytest.skip("Cuaev does not support CPU")
    if device == "cuda" and (not torch.cuda.is_available()):
        pytest.skip("GPU is not available")
    # viral currenly only works for pyaev
    if virial_flag and use_cuaev:
        pytest.skip("virial currently only works for PyAEV")

    # dtype
    dtype = torch.float64 if use_double else torch.float32

    model_loaded = torch.jit.load(modelfile).to(dtype).to(device)
    model_loaded.init(use_cuaev, use_fullnbr)
    use_repulsion = model_loaded.use_repulsion

    model_info = all_models[modelfile]
    if "kwargs" in model_info:
        kwargs = model_info["kwargs"]
    else:
        kwargs = {}
    model_ref_all_models = model_info["model"](**kwargs).to(dtype).to(device)
    model_ref_all_models.set_strategy("cuaev" if use_cuaev else "pyaev")

    if virial_flag and not len(model_ref_all_models.potentials) > 1:
        pytest.skip("we only test virial for simple ANI models")

    # we need a fewer iterations to tigger the fuser
    test_num_models_list = [len(model_ref_all_models.neural_networks)]
    for num_models in test_num_models_list:
        print(f"test num_models == {num_models}")
        model_ref = copy.deepcopy(model_ref_all_models)
        assert isinstance(model_ref.neural_networks, Ensemble)
        model_ref.set_active_members(list(range(num_models)))
        model_loaded.select_models(num_models)
        for i in range(5):
            run_one_test(
                model_ref,
                model_loaded,
                device,
                runpbc,
                use_cuaev,
                use_fullnbr,
                use_repulsion,
                dtype,
                verbose=(i == 0),
                virial_flag=virial_flag,
            )


def run_one_test(
    model_ref,
    model_loaded,
    device,
    runpbc,
    use_cuaev,
    use_fullnbr,
    use_repulsion,
    dtype,
    verbose=False,
    virial_flag=False,
):
    input_file = str(_TESTS_DIR / "water-0.8nm.pdb")
    mol = read(input_file)

    species_periodic_table = torch.tensor(
        mol.get_atomic_numbers(), device=device
    ).unsqueeze(0)
    coordinates = torch.tensor(
        mol.get_positions(), dtype=dtype, requires_grad=True, device=device
    ).unsqueeze(0)
    species = model_ref.species_converter(species_periodic_table)
    cell = torch.tensor(mol.cell.array, device=device, dtype=dtype)
    if not runpbc:
        mol.set_pbc([False, False, False])
    pbc = torch.tensor(mol.pbc, device=device)
    if not runpbc:
        pbc = None
        cell = None
    atom_index12, distances, diff_vector = model_ref.neighborlist(
        model_ref.cutoff, species, coordinates, cell, pbc
    )

    if use_fullnbr:
        ilist_unique, jlist, numneigh = model_ref.aev_computer._half_to_full_nbrlist(
            atom_index12
        )
        para1, para2, para3 = ilist_unique, jlist, numneigh
    else:
        para1, para2, para3 = atom_index12, diff_vector, distances
    species_ghost_as_padding = species.detach().clone()
    torch.set_printoptions(profile="full")

    torch.set_printoptions(precision=13)
    energy, force, _, virial = model_loaded(
        species,
        coordinates,
        para1,
        para2,
        para3,
        species_ghost_as_padding,
        atomic=False,
        virial_flag=virial_flag,
    )

    # test forward_atomic API
    energy_, force_, atomic_energies, virial_ = model_loaded(
        species,
        coordinates,
        para1,
        para2,
        para3,
        species_ghost_as_padding,
        atomic=True,
        virial_flag=virial_flag,
    )

    energy, force, virial = (
        energy * hartree2kcalmol,
        force * hartree2kcalmol,
        virial * hartree2kcalmol,
    )
    energy_, atomic_energies, force_, virial_ = (
        energy_ * hartree2kcalmol,
        atomic_energies * hartree2kcalmol,
        force_ * hartree2kcalmol,
        virial_ * hartree2kcalmol,
    )

    if verbose:
        print(distances.shape)
        print("atomic force max err: ".ljust(15), (force - force_).abs().max().item())
        print(
            f"{'energy:'.ljust(15)} shape: {energy.shape}, value: {energy.item()}, dtype: {energy.dtype}, unit: (kcal/mol)"
        )
        print(
            f"{'force:'.ljust(15)} shape: {force.shape}, dtype: {force.dtype}, unit: (kcal/mol/A)"
        )

    use_double = dtype == torch.float64
    threshold = 1e-13 if use_double else 1.2e-4

    assert torch.allclose(
        energy, energy_, atol=threshold
    ), f"error {(energy - energy_).abs().max()}"
    assert torch.allclose(
        force, force_, atol=threshold
    ), f"error {(force - force_).abs().max()}"
    assert torch.allclose(
        virial, virial_, atol=threshold
    ), f"error {(virial - virial_).abs().max()}"
    # when adding repulsion, atomic_energies does not sum up to total energy
    if not use_repulsion:
        assert torch.allclose(
            energy, atomic_energies.sum(dim=-1), atol=threshold
        ), f"error {(energy - atomic_energies.sum(dim=-1)).abs().max()}"

    # now we run reference calculations
    _, energy_ref = model_ref((species_periodic_table, coordinates), cell, pbc)
    force_ref = -torch.autograd.grad(
        energy_ref.sum(), coordinates, create_graph=True, retain_graph=True
    )[0]
    energy_ref, force_ref = energy_ref * hartree2kcalmol, force_ref * hartree2kcalmol

    # calculate errors
    energy_err = (energy_ref.cpu() - energy.cpu()).abs().max()
    force_err = (force_ref.cpu() - force.cpu()).abs().max()

    if verbose:
        print(
            f"{'energy_ref:'.ljust(15)} shape: {energy_ref.shape}, value: {energy_ref.item()}, dtype: {energy_ref.dtype}, unit: (kcal/mol)"
        )
        print(
            f"{'force_ref:'.ljust(15)} shape: {force_ref.shape}, dtype: {force_ref.dtype}, unit: (kcal/mol/A)"
        )
        print("energy max err: ".ljust(15), energy_err.item())
        print("force  max err: ".ljust(15), force_err.item())

    assert torch.allclose(
        energy, energy_ref
    ), f"error {(energy - energy_ref).abs().max()}"
    assert torch.allclose(
        force, force_ref, atol=threshold
    ), f"error {(force - force_ref).abs().max()}"

    if virial_flag:
        # calculate virial_ref
        calculator = model_ref.ase()
        mol.calc = calculator
        stress = mol.get_stress(voigt=False)
        virial_ref = stress * mol.get_volume() / ase.units.Hartree * hartree2kcalmol
        virial_ref = -virial_ref

        # calculate error
        virial_ref = torch.tensor(virial_ref)
        virial = virial.to(virial_ref.dtype).to(virial_ref.device)
        virial_err = (virial_ref.cpu() - virial.cpu()).abs().max()

        if verbose:
            print("virial max err: ".ljust(15), virial_err.item())

        assert torch.allclose(
            virial, virial_ref, atol=threshold
        ), f"error {(virial - virial_ref).abs().max()}"
