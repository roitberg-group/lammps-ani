import copy
import torch
import pytest
import torchani
# import ani_engine.utils
from ase.io import read
from typing import Tuple, Optional
from torch import Tensor
from torchani.nn import SpeciesEnergies
from torchani.infer import BmmEnsemble2
from torchani.models import Ensemble
from torchani.nn import ANIModel
from torchani.repulsion import RepulsionXTB
from ani2x_ext.custom_emsemble_ani2x_ext import CustomEnsemble

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
hartree2kcalmol = 627.5094738898777

# NVFuser has bug
# torch._C._jit_set_nvfuser_enabled(False)
torch._C._get_graph_executor_optimize(False)


class LammpsModelBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.jit.export
    def init(self, use_cuaev: bool, use_fullnbr: bool):
        """
        Method that will be called at the very beginning within the lammps interface to set parameters.
        """
        raise NotImplementedError

    @torch.jit.export
    def forward(self, species: Tensor, coordinates: Tensor, para1: Tensor, para2: Tensor, para3: Tensor,
                species_ghost_as_padding: Tensor, atomic: bool = False):
        """
        The forward function will be called by the lammps interfact with necessary inputs, and it
        should return 3 tensors: total_energy, atomic_forces, atomic_energies. The atomic_energies could
        be an empty tensor if the `atomic` flag is `False`.

        Args:
            species (Tensor): The species tensor, 0 indexed instead of periodic table indexed.
            coordinates (Tensor): The coordinates tensor.
            para1 (Tensor): if use_fullnbr, it is `ilist_unique`, otherwise `atom_index12`
            para2 (Tensor): if use_fullnbr, it is `jlist`, otherwise `diff_vector`
            para3 (Tensor): if use_fullnbr, it is `numneigh`, otherwise `distances`
            species_ghost_as_padding (Tensor): The species tensor that ghost atoms are set as -1.
            atomic (bool, optional): Whether the atomic_energies should be returned. Defaults to False.

        Raises:
            NotImplementedError: The User needs to override this function.
        """
        raise NotImplementedError

    @torch.jit.export
    def select_models(self, use_num_models: Optional[int] = None):
        self.use_num_models = use_num_models
        """
        For an ensemble of models, select only the first `use_num_models` models.

        Args:
            use_num_models (Optional[int]): Defaults to None.
        """
        pass


########################################################################################

def ANI2x_Model():
    model = torchani.models.ANI2x(periodic_table_index=True, model_index=None, cell_list=False,
                                  use_cuaev_interface=True, use_cuda_extension=True)
    return model


def ANI1x_Zeng():
    eng = ani_engine.utils.load_engine("/blue/roitberg/apps/lammps-ani/myexamples/combustion/retrain_with_zeng/ani_run/logs/debug/20230301_152446-88lx93lb-robust-darkness-5")
    neural_networks = eng.model.networks
    ani1x = torchani.models.ANI1x(periodic_table_index=True, use_cuaev_interface=True, use_cuda_extension=True)
    ani1x.neural_networks = Ensemble([ANIModel(neural_networks)])
    return ani1x


def ANI2x_Repulsion_Model():
    elements = ('H', 'C', 'N', 'O', 'S', 'F', 'Cl')
    def dispersion_atomics(atom: str = 'H'):
        dims_for_atoms = {'H': (1008, 256, 192, 160),
                          'C': (1008, 256, 192, 160),
                          'N': (1008, 192, 160, 128),
                          'O': (1008, 192, 160, 128),
                          'S': (1008, 160, 128, 96),
                          'F': (1008, 160, 128, 96),
                          'Cl': (1008, 160, 128, 96)}
        return torchani.atomics.standard(dims_for_atoms[atom], activation=torch.nn.GELU(), bias=False)
    model = torchani.models.ANI2x(pretrained=False,
                  cutoff_fn='smooth',
                  atomic_maker=dispersion_atomics,
                  ensemble_size=7,
                  repulsion=True,
                  repulsion_kwargs={'symbols': elements,
                                    'cutoff': 5.1,
                                    'cutoff_fn': torchani.aev.cutoffs.CutoffSmooth(order=2)},
                                    periodic_table_index=True, model_index=None, cell_list=False,
                                    use_cuaev_interface=True, use_cuda_extension=True
                  )
    state_dict = torchani.models._fetch_state_dict('anid_state_dict_mod.pt', private=True)
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
    return model


# class ANI2xExt_Model(CustomEnsemble):
#     """
#     ani_ext model with repulsion, smooth cutoff, GELU, No Bias, GSAE
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.aev_computer = torchani.AEVComputer.like_2x(cutoff_fn="smooth", use_cuda_extension=True, use_cuaev_interface=True)
#         self.neural_networks = self.models

#     def forward(self):
#         pass

########################################################################################


class LammpsANI(LammpsModelBase):
    def __init__(self, model, use_repulsion):
        super().__init__()

        # setup model
        self.use_cuaev = True
        self.use_fullnbr = True
        self.initialized = False
        self.use_repulsion = use_repulsion

        assert hasattr(model, 'aev_computer'), "No aev_computer found in the model"
        assert hasattr(model, 'neural_networks'), "No neural_networks found in the model"
        assert isinstance(model.neural_networks, Ensemble) or isinstance(model.neural_networks, torch.nn.ModuleList)
        assert hasattr(model, 'energy_shifter'), "No energy_shifter found in the model"

        self.aev_computer = model.aev_computer
        # TODO how to set repulsion cutoff
        # TODO how to set repulsion elements
        # TODO how to support general model
        elements = ('H', 'C', 'N', 'O', 'S', 'F', 'Cl')
        # elements = ('H', 'C', 'N', 'O')
        self.rep_calc = RepulsionXTB(cutoff=5.1, symbols=elements)

        # num_models
        self.num_models = len(model.neural_networks)
        self.use_num_models = self.num_models
        # Batched neural networks is required for selecting number of models at Runtime.
        # TODO if the normal Ensemble needs to be supported to select_models in the future,
        # A ModuleList of Ensemble with different number of models could be prepared in advance
        # within the __init__ function.
        self.neural_networks = BmmEnsemble2(model.neural_networks)
        self.energy_shifter = model.energy_shifter
        self.register_buffer("dummy_buffer", torch.empty(0))
        # self.nvfuser_enabled = torch._C._jit_nvfuser_enabled()

        # we don't need weight gradient when calculating force
        for name, param in self.neural_networks.named_parameters():
            param.requires_grad_(False)

        self.using_bmmensemble = isinstance(self.neural_networks, BmmEnsemble2)

    @torch.jit.export
    def init(self, use_cuaev: bool, use_fullnbr: bool):
        self.use_cuaev = use_cuaev
        self.use_fullnbr = use_fullnbr
        self.initialized = True

    @torch.jit.export
    def forward(self, species: Tensor, coordinates: Tensor, para1: Tensor, para2: Tensor, para3: Tensor,
                species_ghost_as_padding: Tensor, atomic: bool=False):
        assert self.initialized, "Model is not initialized, You need to call init() method before forward function"

        if self.use_cuaev and not self.aev_computer.cuaev_is_initialized:
            self.aev_computer._init_cuaev_computer()
            self.aev_computer.cuaev_is_initialized = True
        # when use ghost_index and mnp, the input system must be a single molecule

        torch.ops.mnp.nvtx_range_push("AEV forward")
        aev = self.compute_aev(species, coordinates, para1, para2, para3)
        torch.ops.mnp.nvtx_range_pop()

        if atomic:
            energies, atomic_energies = self.forward_atomic(species, coordinates, species_ghost_as_padding, aev)
        else:
            energies, atomic_energies = self.forward_total(species, coordinates, species_ghost_as_padding, aev)

        if self.use_repulsion:
            torch.ops.mnp.nvtx_range_push("Repulsion forward")
            ghost_flags = (species_ghost_as_padding == -1)
            rep_energies = self.compute_repulsion(species, coordinates, para1, para2, para3, ghost_flags)
            energies += rep_energies
            torch.ops.mnp.nvtx_range_pop()

        torch.ops.mnp.nvtx_range_push("Force")
        force = torch.autograd.grad([energies.sum()], [coordinates], create_graph=True, retain_graph=True)[0]
        assert force is not None
        force = -force
        torch.ops.mnp.nvtx_range_pop()

        return energies, force, atomic_energies

    @torch.jit.export
    def forward_total(self, species: Tensor, coordinates: Tensor, species_ghost_as_padding: Tensor, aev: Tensor):
        # run neural networks
        torch.ops.mnp.nvtx_range_push(f"NN ({self.use_num_models}) forward")
        species_energies = self.neural_networks((species_ghost_as_padding, aev))
        # TODO force is independent of energy_shifter?
        species_energies = self.energy_shifter(species_energies)
        energies = species_energies[1]
        torch.ops.mnp.nvtx_range_pop()

        return energies, torch.empty(0)

    @torch.jit.export
    def forward_atomic(self, species: Tensor, coordinates: Tensor, species_ghost_as_padding: Tensor, aev: Tensor):
        ntotal = species.shape[1]
        nghost = (species_ghost_as_padding == -1).flatten().sum()
        nlocal = ntotal - nghost

        # run neural networks
        torch.ops.mnp.nvtx_range_push("NN ({self.use_num_models}) forward_atomic")
        atomic_energies = self.neural_networks._atomic_energies((species_ghost_as_padding, aev))
        atomic_energies += self.energy_shifter._atomic_saes(species_ghost_as_padding)
        # when using ANI ensemble (not batchmm), atomic_energies shape is [models, C, A]
        if len(atomic_energies.shape) > 2:
            atomic_energies = atomic_energies.mean(0)
        energies = atomic_energies.sum(dim=1)
        torch.ops.mnp.nvtx_range_pop()

        return energies, atomic_energies[:, :nlocal]

    @torch.jit.export
    def compute_aev(self, species: Tensor, coordinates: Tensor, para1: Tensor, para2: Tensor, para3: Tensor):
        atom_index12, diff_vector, distances = para1, para2, para3
        ilist_unique, jlist, numneigh = para1, para2, para3
        # compute aev
        assert species.shape[0] == 1, "Currently only support inference for single molecule"
        if self.use_cuaev:
            if self.use_fullnbr:
                aev = self.aev_computer._compute_cuaev_with_full_nbrlist(species, coordinates, ilist_unique, jlist, numneigh)
            else:
                aev = self.aev_computer._compute_cuaev_with_half_nbrlist(species, coordinates, atom_index12, diff_vector, distances)
            assert aev is not None
        else:
            # diff_vector, distances from lammps are always in double,
            # we need to convert it to single precision if needed
            if self.use_fullnbr:
                atom_index12 = self.aev_computer._full_to_half_nbrlist(ilist_unique, jlist, numneigh, species)
                # print(f"{atom_index12.device}, max_neighbor_index {atom_index12.max().item()}, num_atoms {coordinates.shape[1]}")
                assert atom_index12.max() < coordinates.shape[1], f"neighbor {atom_index12.max().item()} larger than num_atoms {coordinates.shape[1]}"
                coords0 = coordinates.view(-1, 3).index_select(0, atom_index12[0])
                coords1 = coordinates.view(-1, 3).index_select(0, atom_index12[1])
                diff_vector = coords0 - coords1
                distances = diff_vector.norm(2, -1)
            aev = self.aev_computer._compute_aev(species, atom_index12, distances, diff_vector)

        return aev

    @torch.jit.export
    def compute_repulsion(self, species: Tensor, coordinates: Tensor, para1: Tensor, para2: Tensor, para3: Tensor, ghost_flags: Tensor):
        atom_index12, diff_vector, distances = para1, para2, para3
        ilist_unique, jlist, numneigh = para1, para2, para3
        if self.use_fullnbr:
            atom_index12 = self.aev_computer._full_to_half_nbrlist(ilist_unique, jlist, numneigh, species)
            assert atom_index12.max() < coordinates.shape[1], f"neighbor {atom_index12.max().item()} larger than num_atoms {coordinates.shape[1]}"
            coords0 = coordinates.view(-1, 3).index_select(0, atom_index12[0])
            coords1 = coordinates.view(-1, 3).index_select(0, atom_index12[1])
            diff_vector = coords0 - coords1
            distances = diff_vector.norm(2, -1)
        repulsion_energies = self.rep_calc(species, atom_index12, distances, ghost_flags=ghost_flags)
        return repulsion_energies

    @torch.jit.export
    def select_models(self, use_num_models: Optional[int] = None):
        if self.using_bmmensemble:
            self.neural_networks.select_models(use_num_models)
            self.use_num_models = self.neural_networks.use_num_models
        elif use_num_models is None or use_num_models == self.num_models:
            # We don't need to do anything in this case, even if it is not using BmmEnsemble2.
            pass
        else:
            raise RuntimeError("select_models method only works for BmmEnsemble2")


# TODO delete this
class ANI2xRef(torch.nn.Module):
    def __init__(self, use_cuaev, use_repulsion):
        super().__init__()
        ani2x = torchani.models.ANI2x(periodic_table_index=True, model_index=None, cell_list=False,
                                      use_cuaev_interface=use_cuaev, use_cuda_extension=use_cuaev)
        self.model = ani2x
        self.model.neural_networks = self.model.neural_networks.to_infer_model(use_mnp=False)
        self.use_cuaev = use_cuaev
        # we only use halfnbr for ANI2xRef
        # self.use_fullnbr = use_fullnbr
        # self.model.aev_computer.use_fullnbr = use_fullnbr
        self.use_repulsion = use_repulsion
        self.rep_calc = RepulsionXTB()
        self.periodic_table_index = self.model.periodic_table_index
        self.aev_computer = self.model.aev_computer

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        species_coordinates = self.model._maybe_convert_species(species_coordinates)
        species_aevs = self.model.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.model.neural_networks(species_aevs)
        energies = self.model.energy_shifter(species_energies).energies
        if self.use_repulsion:
            energies += self.compute_repulsion(species_coordinates, cell, pbc)
        return SpeciesEnergies(species_coordinates[0], energies)

    def compute_repulsion(self, species_coordinates: Tuple[Tensor, Tensor],
                          cell: Optional[Tensor] = None, pbc: Optional[Tensor] = None):
        species, coordinates = species_coordinates
        atom_index12, distances, _, _ = self.model.aev_computer.neighborlist(species, coordinates, cell, pbc)
        return self.rep_calc(species, atom_index12, distances)

    @torch.jit.export
    def select_models(self, use_num_models: Optional[int] = None):
        neural_networks = self.model.neural_networks
        if isinstance(neural_networks, BmmEnsemble2):
            neural_networks.select_models(use_num_models)
        elif isinstance(neural_networks, Ensemble):
            size = len(neural_networks)
            if use_num_models is None:
                use_num_models = size
                return
            assert use_num_models <= size, f"use_num_models {use_num_models} cannot be larger than size {size}"
            neural_networks = neural_networks[:use_num_models]
        else:
            raise RuntimeError("select_models method only works for BmmEnsemble2 or Ensemble neural networks")


all_models = {"ani2x.pt": {"model": ANI2x_Model, "use_repulsion": False},
              "ani2x_repulsion.pt": {"model": ANI2x_Repulsion_Model, "use_repulsion": True},
            #   "ani1x_zeng.pt": {"model": ANI1x_Zeng, "use_repulsion": True},
            #   "ani2x_ext0_repulsion": {"model": ANI2xExt_Model, "use_repulsion": True},
              }

def save_ani2x_model():
    for output_file, info in all_models.items():
        ani2x = LammpsANI(info["model"](), use_repulsion=info["use_repulsion"])
        script_module = torch.jit.script(ani2x)
        script_module.save(output_file)


# Save all ani2x models by using session-scoped "autouse" fixture, this will run ahead of all tests.
@pytest.fixture(scope='session', autouse=True)
def session_start():
    print('Pytest session started, saving all models')
    save_ani2x_model()

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

@pytest.mark.parametrize("runpbc", runpbc_params)
@pytest.mark.parametrize("device", device_params)
@pytest.mark.parametrize("use_double", use_double_params)
@pytest.mark.parametrize("use_cuaev", use_cuaev_params)
@pytest.mark.parametrize("use_fullnbr", use_fullnbr_params)
@pytest.mark.parametrize("modelfile", modelfile_params)
def test_ani2x_models(runpbc, device, use_double, use_cuaev, use_fullnbr, modelfile):
    # when pbc is on, full nbrlist converted from half nbrlist is not correct
    if use_fullnbr and runpbc:
        pytest.skip("Does not support full neighbor list using pyaev when pbc is on")
    if use_cuaev and device == 'cpu':
        pytest.skip("Cuaev does not support CPU")
    if device == 'cuda' and (not torch.cuda.is_available()):
        pytest.skip("GPU is not available")

    # dtype
    dtype = torch.float64 if use_double else torch.float32
    use_repulsion = all_models[modelfile]["use_repulsion"]

    # cuaev currently only works with single precision
    ani2x_loaded = torch.jit.load(modelfile).to(dtype).to(device)
    # ani2x_loaded = LammpsANI(ANI2x_Repulsion_Model(), use_repulsion=use_repulsion).to(dtype).to(device)
    ani2x_loaded.init(use_cuaev, use_fullnbr)

    def set_cuda_aev(model, use_cuaev):
        model.aev_computer.use_cuaev_interface = use_cuaev
        model.aev_computer.use_cuda_extension = use_cuaev
        return model

    def select_num_models(model, use_num_models):
        newmodel = copy.deepcopy(model)
        # BuiltinModelPairInteractions needs to change AEVPotential.neural_networks
        if isinstance(model, torchani.models.BuiltinModelPairInteractions):
            assert isinstance(model.neural_networks, torch.nn.ModuleList)
            assert use_num_models <= len(model.neural_networks)
            newmodel.neural_networks = Ensemble(newmodel.neural_networks[:use_num_models])
            for i, pot in enumerate(model.potentials):
                if isinstance(pot, torchani.models.AEVPotential):
                    nn = newmodel.potentials[i].neural_networks
                    assert isinstance(nn, torch.nn.ModuleList)
                    assert use_num_models <= len(nn)
                    newmodel.potentials[i].neural_networks = Ensemble(nn[:use_num_models])
        elif isinstance(model, torchani.models.BuiltinModel):
            assert isinstance(model.neural_networks, torch.nn.ModuleList)
            assert use_num_models <= len(model.neural_networks)
            newmodel.neural_networks = Ensemble(newmodel.neural_networks[:use_num_models])
        return newmodel

    ani2x_ref_all_models = all_models[modelfile]["model"]().to(dtype).to(device)
    ani2x_ref_all_models = set_cuda_aev(ani2x_ref_all_models, use_cuaev)

    # ani2x_ref = ANI2xRef(use_cuaev, use_repulsion).to(dtype).to(device)

    # we need a fewer iterations to tigger the fuser
    total_num_models = len(ani2x_ref_all_models.neural_networks)
    for num_models in [total_num_models]:
        ani2x_ref = ani2x_ref_all_models
        ani2x_loaded.select_models(num_models)
        for i in range(5):
            run_one_test(ani2x_ref, ani2x_loaded, device, runpbc, use_cuaev, use_fullnbr, use_repulsion, dtype, verbose=(num_models == total_num_models and i==0))


def run_one_test(ani2x_ref, ani2x_loaded, device, runpbc, use_cuaev, use_fullnbr, use_repulsion, dtype, verbose=False):
    input_file = "water-0.8nm.pdb"
    mol = read(input_file)

    species_periodic_table = torch.tensor(mol.get_atomic_numbers(), device=device).unsqueeze(0)
    coordinates = torch.tensor(mol.get_positions(), dtype=dtype, requires_grad=True, device=device).unsqueeze(0)
    species, coordinates = ani2x_ref.species_converter((species_periodic_table, coordinates))
    cell = torch.tensor(mol.cell.array, device=device, dtype=dtype)
    pbc = torch.tensor(mol.pbc, device=device)

    if runpbc:
        atom_index12, distances, diff_vector, _ = ani2x_ref.aev_computer.neighborlist(species, coordinates, cell, pbc)
    else:
        atom_index12, distances, diff_vector, _ = ani2x_ref.aev_computer.neighborlist(species, coordinates)
    if use_fullnbr:
        ilist_unique, jlist, numneigh = ani2x_ref.aev_computer._half_to_full_nbrlist(atom_index12)
        para1, para2, para3 = ilist_unique, jlist, numneigh
    else:
        para1, para2, para3 = atom_index12, diff_vector, distances
    species_ghost_as_padding = species.detach().clone()
    torch.set_printoptions(profile="full")

    torch.set_printoptions(precision=13)
    energy, force, _ = ani2x_loaded(species, coordinates, para1, para2, para3, species_ghost_as_padding, atomic=False)

    # test forward_atomic API
    energy_, force_, atomic_energies = ani2x_loaded(species, coordinates, para1, para2, para3, species_ghost_as_padding, atomic=True)

    energy, force = energy * hartree2kcalmol, force * hartree2kcalmol
    energy_, atomic_energies, force_ = energy_ * hartree2kcalmol, atomic_energies * hartree2kcalmol, force_ * hartree2kcalmol

    if verbose:
        print(distances.shape)
        print("atomic force max err: ".ljust(15), (force - force_).abs().max().item())
        print(f"{'energy:'.ljust(15)} shape: {energy.shape}, value: {energy.item()}, dtype: {energy.dtype}, unit: (kcal/mol)")
        print(f"{'force:'.ljust(15)} shape: {force.shape}, dtype: {force.dtype}, unit: (kcal/mol/A)")

    use_double = dtype == torch.float64
    threshold = 1e-13 if use_double else 1e-4

    assert torch.allclose(energy, energy_, atol=threshold), f"error {(energy - energy_).abs().max()}"
    assert torch.allclose(force, force_, atol=threshold), f"error {(force - force_).abs().max()}"
    if not use_repulsion:
        assert torch.allclose(energy, atomic_energies.sum(dim=-1), atol=threshold), f"error {(energy - atomic_energies.sum(dim=-1)).abs().max()}"

    # for test_model inputs
    # print(coordinates.flatten())
    # print(species.flatten())
    # print(atom_index12.flatten())
    # print(force.flatten())

    if runpbc:
        _, energy_ref = ani2x_ref((species_periodic_table, coordinates), cell, pbc)
    else:
        _, energy_ref = ani2x_ref((species_periodic_table, coordinates))
    force_ref = -torch.autograd.grad(energy_ref.sum(), coordinates, create_graph=True, retain_graph=True)[0]
    energy_ref, force_ref = energy_ref * hartree2kcalmol, force_ref * hartree2kcalmol

    energy_err = (energy_ref.cpu() - energy.cpu()).abs().max()
    force_err = (force_ref.cpu() - force.cpu()).abs().max()

    if verbose:
        print(f"{'energy_ref:'.ljust(15)} shape: {energy_ref.shape}, value: {energy_ref.item()}, dtype: {energy_ref.dtype}, unit: (kcal/mol)")
        print(f"{'force_ref:'.ljust(15)} shape: {force_ref.shape}, dtype: {force_ref.dtype}, unit: (kcal/mol/A)")
        print("energy max err: ".ljust(15), energy_err.item())
        print("force  max err: ".ljust(15), force_err.item())

    # print(f"error {(energy - energy_ref).abs().max()}")
    assert torch.allclose(energy, energy_ref), f"error {(energy - energy_ref).abs().max()}"
    assert torch.allclose(force, force_ref, atol=threshold), f"error {(force - force_ref).abs().max()}"
