import copy
import torch
import pytest
import torchani
from ase.io import read
from torchani.models import Ensemble
from .models import all_models, save_models

hartree2kcalmol = torchani.units.hartree2kcalmol(1)

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
modelfile_params = all_models.keys()
# remove modelfiles that have unittest as False
modelfile_params = [
    modelfile for modelfile in modelfile_params if all_models[modelfile]["unittest"]
]


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
def test_models(runpbc, device, use_double, use_cuaev, use_fullnbr, modelfile):
    # when pbc is on, full nbrlist converted from half nbrlist is not correct
    if use_fullnbr and runpbc:
        pytest.skip("Does not support full neighbor list using pyaev when pbc is on")
    if use_cuaev and device == "cpu":
        pytest.skip("Cuaev does not support CPU")
    if device == "cuda" and (not torch.cuda.is_available()):
        pytest.skip("GPU is not available")

    # dtype
    dtype = torch.float64 if use_double else torch.float32

    model_loaded = torch.jit.load(modelfile).to(dtype).to(device)
    model_loaded.init(use_cuaev, use_fullnbr)
    use_repulsion = model_loaded.use_repulsion

    def set_ref_cuda_aev(model, use_cuaev):
        model.aev_computer.use_cuaev_interface = use_cuaev
        model.aev_computer.use_cuda_extension = use_cuaev
        return model

    def select_ref_num_models(model, use_num_models):
        newmodel = copy.deepcopy(model)
        # BuiltinModelPairInteractions needs to change AEVPotential.neural_networks
        if isinstance(model, torchani.models.BuiltinModelPairInteractions):
            assert isinstance(model.neural_networks, torch.nn.ModuleList)
            assert use_num_models <= len(model.neural_networks)
            newmodel.neural_networks = Ensemble(
                newmodel.neural_networks[:use_num_models]
            )
            for i, pot in enumerate(model.potentials):
                if isinstance(pot, torchani.models.AEVPotential):
                    nn = newmodel.potentials[i].neural_networks
                    assert isinstance(nn, torch.nn.ModuleList)
                    assert use_num_models <= len(nn)
                    newmodel.potentials[i].neural_networks = Ensemble(
                        nn[:use_num_models]
                    )
        elif isinstance(model, torchani.models.BuiltinModel):
            assert isinstance(model.neural_networks, torch.nn.ModuleList)
            assert use_num_models <= len(model.neural_networks)
            newmodel.neural_networks = Ensemble(
                newmodel.neural_networks[:use_num_models]
            )
        return newmodel

    model_ref_all_models = all_models[modelfile]["model"]().to(dtype).to(device)
    model_ref_all_models = set_ref_cuda_aev(model_ref_all_models, use_cuaev)

    # we need a fewer iterations to tigger the fuser
    total_num_models = len(model_ref_all_models.neural_networks)
    test_num_models_list = []
    if total_num_models <= 4:
        test_num_models_list = [total_num_models]
    else:
        test_num_models_list = [4, total_num_models]
    for num_models in test_num_models_list:
        print(f"test num_models == {num_models}")
        model_ref = select_ref_num_models(model_ref_all_models, num_models)
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
):
    input_file = "water-0.8nm.pdb"
    mol = read(input_file)

    species_periodic_table = torch.tensor(
        mol.get_atomic_numbers(), device=device
    ).unsqueeze(0)
    coordinates = torch.tensor(
        mol.get_positions(), dtype=dtype, requires_grad=True, device=device
    ).unsqueeze(0)
    species, coordinates = model_ref.species_converter(
        (species_periodic_table, coordinates)
    )
    cell = torch.tensor(mol.cell.array, device=device, dtype=dtype)
    pbc = torch.tensor(mol.pbc, device=device)

    if runpbc:
        atom_index12, distances, diff_vector, _ = model_ref.aev_computer.neighborlist(
            species, coordinates, cell, pbc
        )
    else:
        atom_index12, distances, diff_vector, _ = model_ref.aev_computer.neighborlist(
            species, coordinates
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
    energy, force, _ = model_loaded(
        species,
        coordinates,
        para1,
        para2,
        para3,
        species_ghost_as_padding,
        atomic=False,
    )

    # test forward_atomic API
    energy_, force_, atomic_energies = model_loaded(
        species, coordinates, para1, para2, para3, species_ghost_as_padding, atomic=True
    )

    energy, force = energy * hartree2kcalmol, force * hartree2kcalmol
    energy_, atomic_energies, force_ = (
        energy_ * hartree2kcalmol,
        atomic_energies * hartree2kcalmol,
        force_ * hartree2kcalmol,
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
    threshold = 1e-13 if use_double else 1e-4

    assert torch.allclose(
        energy, energy_, atol=threshold
    ), f"error {(energy - energy_).abs().max()}"
    assert torch.allclose(
        force, force_, atol=threshold
    ), f"error {(force - force_).abs().max()}"
    if not use_repulsion:
        assert torch.allclose(
            energy, atomic_energies.sum(dim=-1), atol=threshold
        ), f"error {(energy - atomic_energies.sum(dim=-1)).abs().max()}"

    if runpbc:
        _, energy_ref = model_ref((species_periodic_table, coordinates), cell, pbc)
    else:
        _, energy_ref = model_ref((species_periodic_table, coordinates))
    force_ref = -torch.autograd.grad(
        energy_ref.sum(), coordinates, create_graph=True, retain_graph=True
    )[0]
    energy_ref, force_ref = energy_ref * hartree2kcalmol, force_ref * hartree2kcalmol

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
