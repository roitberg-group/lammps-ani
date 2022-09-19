import torch
import torchani
from ase.io import read
import argparse
from typing import Tuple, Optional
from torch import Tensor
from torchani.nn import SpeciesEnergies
from torchani.infer import BmmEnsemble2
from torchani.models import Ensemble

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
hartree2kcalmol = 627.5094738898777

# NVFuser has bug
# torch._C._jit_set_nvfuser_enabled(False)
torch._C._get_graph_executor_optimize(False)


class ANI2x(torch.nn.Module):
    def __init__(self, use_cuaev, use_fullnbr):
        super().__init__()
        ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=None, cell_list=False,
                                      use_cuaev_interface=use_cuaev, use_cuda_extension=use_cuaev,
                                      use_fullnbr=use_fullnbr)
        self.use_cuaev = use_cuaev
        self.aev_computer = ani2x.aev_computer
        # num_models
        self.num_models = len(ani2x.neural_networks)
        self.use_num_models = self.num_models
        # batched neural networks
        self.neural_networks = ani2x.neural_networks.to_infer_model(use_mnp=False)
        # self.neural_networks = ani2x.neural_networks
        self.energy_shifter = ani2x.energy_shifter
        self.register_buffer("dummy_buffer", torch.empty(0))
        self.use_fullnbr = use_fullnbr
        # self.nvfuser_enabled = torch._C._jit_nvfuser_enabled()

        # we don't need weight gradient when calculating force
        for name, param in self.neural_networks.named_parameters():
            param.requires_grad_(False)

    @torch.jit.export
    def forward(self, species, coordinates, para1, para2, para3, species_ghost_as_padding, atomic: bool=False):
        if self.use_cuaev and not self.aev_computer.cuaev_is_initialized:
            self.aev_computer._init_cuaev_computer()
            self.aev_computer.cuaev_is_initialized = True
        # when use ghost_index and mnp, the input system must be a single molecule

        torch.ops.mnp.nvtx_range_push("AEV forward")
        aev = self.compute_aev(species, coordinates, para1, para2, para3)
        torch.ops.mnp.nvtx_range_pop()
        if atomic:
            return self.forward_atomic(species, coordinates, species_ghost_as_padding, aev)
        else:
            return self.forward_total(species, coordinates, species_ghost_as_padding, aev)

    @torch.jit.export
    def forward_total(self, species, coordinates, species_ghost_as_padding, aev):
        # run neural networks
        torch.ops.mnp.nvtx_range_push("NN forward")
        species_energies = self.neural_networks((species_ghost_as_padding, aev))
        # TODO force is independent of energy_shifter?
        species_energies = self.energy_shifter(species_energies)
        energies = species_energies[1]
        torch.ops.mnp.nvtx_range_pop()

        torch.ops.mnp.nvtx_range_push("Force")
        force = torch.autograd.grad([energies.sum()], [coordinates], create_graph=True, retain_graph=True)[0]
        assert force is not None
        force = -force
        torch.ops.mnp.nvtx_range_pop()

        return energies, force, torch.empty(0)

    @torch.jit.export
    def forward_atomic(self, species, coordinates, species_ghost_as_padding, aev):
        ntotal = species.shape[1]
        nghost = (species_ghost_as_padding == -1).flatten().sum()
        nlocal = ntotal - nghost

        # run neural networks
        torch.ops.mnp.nvtx_range_push("NN forward")
        atomic_energies = self.neural_networks._atomic_energies((species_ghost_as_padding, aev))
        atomic_energies += self.energy_shifter._atomic_saes(species_ghost_as_padding)
        energies = atomic_energies.sum(dim=1)
        torch.ops.mnp.nvtx_range_pop()

        torch.ops.mnp.nvtx_range_push("Force")
        force = torch.autograd.grad([energies.sum()], [coordinates], create_graph=True, retain_graph=True)[0]
        assert force is not None
        force = -force
        torch.ops.mnp.nvtx_range_pop()

        return energies, force, atomic_energies[:, :nlocal]

    @torch.jit.export
    def compute_aev(self, species, coordinates, para1, para2, para3):
        # dtype
        dtype = self.dummy_buffer.dtype

        atom_index12, diff_vector, distances = para1, para2, para3
        ilist_unique, jlist, numneigh = para1, para2, para3
        # compute aev
        assert species.shape[0] == 1, "Currently only support inference for single molecule"
        if self.use_cuaev:
            # diff_vector, distances, coordinates from lammps are always in double,
            # cuaev currently only works with single precision
            diff_vector = diff_vector.to(torch.float32)
            distances = distances.to(torch.float32)
            coordinates = coordinates.to(torch.float32)
            if self.use_fullnbr:
                aev = self.aev_computer._compute_cuaev_with_full_nbrlist(species, coordinates, ilist_unique, jlist, numneigh)
            else:
                aev = self.aev_computer._compute_cuaev_with_nbrlist(species, coordinates, atom_index12, diff_vector, distances)
            assert aev is not None
            # the neural network part will use whatever dtype the user specified
            aev = aev.to(dtype)
        else:
            # diff_vector, distances from lammps are always in double,
            # we need to convert it to single precision if needed
            diff_vector = diff_vector.to(dtype)
            distances = distances.to(dtype)
            aev = self.aev_computer._compute_aev(species, atom_index12, diff_vector, distances)

        return aev

    @torch.jit.export
    def select_models(self, use_num_models: Optional[int] = None):
        neural_networks = self.neural_networks
        if isinstance(neural_networks, BmmEnsemble2):
            neural_networks.select_models(use_num_models)
            self.use_num_models = neural_networks.use_num_models
        elif isinstance(neural_networks, Ensemble):
            size = len(neural_networks)
            if use_num_models is None:
                use_num_models = size
                return
            assert use_num_models <= size, f"use_num_models {use_num_models} cannot be larger than size {size}"
            neural_networks = neural_networks[:use_num_models]
            self.use_num_models = use_num_models
        else:
            raise RuntimeError("select_models method only works for BmmEnsemble2 or Ensemble neural networks")


class ANI2xRef(torch.nn.Module):
    """
    This is used to handel cuaev_computer that currently only works with single precision.
    """
    def __init__(self, use_cuaev, use_fullnbr):
        super().__init__()
        ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=None, cell_list=False,
                                      use_cuaev_interface=use_cuaev, use_cuda_extension=use_cuaev,
                                      use_fullnbr=use_fullnbr)
        self.model = ani2x
        self.model.neural_networks = self.model.neural_networks.to_infer_model(use_mnp=False)
        self.use_cuaev = use_cuaev
        self.register_buffer("dummy_buffer", torch.empty(0))
        self.use_fullnbr = use_fullnbr

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        species_coordinates = self.model._maybe_convert_species(species_coordinates)
        if self.use_cuaev:
            species_coordinates = (species_coordinates[0], species_coordinates[1].to(torch.float32))
            if cell is not None:
                cell = cell.to(torch.float32)
            species_aevs = self.model.aev_computer(species_coordinates, cell=cell, pbc=pbc)
            species_aevs = (species_aevs[0], species_aevs[1].to(self.dummy_buffer.dtype))
        else:
            species_aevs = self.model.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.model.neural_networks(species_aevs)
        return self.model.energy_shifter(species_energies)

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


def save_ani2x_model(runpbc=False, device='cuda', use_double=True, use_cuaev=False, use_fullnbr=False):
    # dtype
    dtype = torch.float64 if use_double else torch.float32

    ani2x = ANI2x(use_cuaev, use_fullnbr)
    ani2x = ani2x.to(dtype)
    # cuaev currently only works with single precision
    if use_cuaev:
        ani2x.aev_computer = ani2x.aev_computer.to(torch.float32)
    script_module = torch.jit.script(ani2x)
    script_module.save(output_file)

    ani2x_loaded = torch.jit.load(output_file).to(device)
    # ani2x_loaded = ani2x.to(device)
    ani2x_ref = ANI2xRef(use_cuaev, use_fullnbr).to(device).to(dtype)
    # cuaev currently only works with single precision
    if use_cuaev:
        ani2x_ref.model.aev_computer = ani2x_ref.model.aev_computer.to(torch.float32)

    # we need a fewer iterations to tigger the fuser
    for num_models in [None, 4]:
        ani2x_ref.select_models(num_models)
        ani2x_loaded.select_models(num_models)
        for i in range(5):
            test(ani2x_ref, ani2x_loaded, device, runpbc, use_cuaev, use_fullnbr, dtype, verbose=(num_models is None and i==0))


def test(ani2x_ref, ani2x_loaded, device, runpbc, use_cuaev, use_fullnbr, dtype, verbose=False):
    mol = read(input_file)

    species = torch.tensor(mol.get_atomic_numbers(), device=device).unsqueeze(0)
    coordinates = torch.tensor(mol.get_positions(), dtype=dtype, requires_grad=True, device=device).unsqueeze(0)
    species, coordinates = ani2x_ref.model.species_converter((species, coordinates))
    cell = torch.tensor(mol.cell, device=device, dtype=dtype)
    pbc = torch.tensor(mol.pbc, device=device)

    # TODO It is IMPORTANT to set cutoff as 7.1 to match lammps nbr cutoff
    ani2x_ref.model.aev_computer.neighborlist.cutoff = 7.1
    if use_cuaev:
        coordinates = coordinates.to(torch.float32)
        cell = cell.to(torch.float32)
    if runpbc:
        atom_index12, _, diff_vector, distances = ani2x_ref.model.aev_computer.neighborlist(species, coordinates, cell, pbc)
    else:
        atom_index12, _, diff_vector, distances = ani2x_ref.model.aev_computer.neighborlist(species, coordinates)
    if use_fullnbr:
        ilist_unique, jlist, numneigh = ani2x_ref.model.aev_computer._half_to_full_nbrlist(atom_index12)
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

    if use_cuaev:
        threshold = 9.5e-5
    else:
        threshold = 1e-7 if use_double else 3e-5

    assert torch.allclose(energy, energy_, atol=threshold), f"error {(energy - energy_).abs().max()}"
    assert torch.allclose(force, force_, atol=threshold), f"error {(force - force_).abs().max()}"
    assert torch.allclose(energy, atomic_energies.sum(dim=-1), atol=threshold), f"error {(energy - atomic_energies.sum(dim=-1)).abs().max()}"

    # for test_model inputs
    # print(coordinates.flatten())
    # print(species.flatten())
    # print(atom_index12.flatten())
    # print(force.flatten())

    if runpbc:
        _, energy_ref = ani2x_ref((species, coordinates), cell, pbc)
    else:
        _, energy_ref = ani2x_ref((species, coordinates))
    force_ref = -torch.autograd.grad(energy_ref.sum(), coordinates, create_graph=True, retain_graph=True)[0]
    energy_ref, force_ref = energy_ref * hartree2kcalmol, force_ref * hartree2kcalmol

    energy_err = torch.abs(torch.max(energy_ref.cpu() - energy.cpu()))
    force_err = torch.abs(torch.max(force_ref.cpu() - force.cpu()))

    if verbose:
        print(f"{'energy_ref:'.ljust(15)} shape: {energy_ref.shape}, value: {energy_ref.item()}, dtype: {energy_ref.dtype}, unit: (kcal/mol)")
        print(f"{'force_ref:'.ljust(15)} shape: {force_ref.shape}, dtype: {force_ref.dtype}, unit: (kcal/mol/A)")
        print("energy max err: ".ljust(15), energy_err.item())
        print("force  max err: ".ljust(15), force_err.item())

    assert torch.allclose(energy, energy_ref, atol=threshold), f"error {(energy - energy_ref).abs().max()}"
    assert torch.allclose(force, force_ref, atol=threshold), f"error {(force - force_ref).abs().max()}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pbc', default=False, action='store_true')
    args = parser.parse_args()
    input_file = "water-0.8nm.pdb"

    devices = ['cpu']
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            # avoid the bug that could only use the 0th gpu
            devices.append('cuda:1')
        else:
            devices.append('cuda:0')

    for use_cuaev in [True, False]:
        full_nbrlist = [True, False] if use_cuaev else [False]
        for use_fullnbr in full_nbrlist:
            full_or_half = "full" if use_fullnbr else "half"
            cuaev_or_nocuaev = "cuaev" if use_cuaev else "nocuaev"
            for use_double in [True, False]:
                double_or_single = "double" if use_double else "single"
                output_file = f'ani2x_{cuaev_or_nocuaev}_{double_or_single}_{full_or_half}.pt'
                print(output_file)
                for pbc in [False, True]:
                    for d in devices:
                        if use_cuaev and d == "cpu":
                            continue
                        print(f"====================== {cuaev_or_nocuaev} | {full_or_half} | {double_or_single} | pbc: {pbc} | device: {d} ======================")
                        save_ani2x_model(pbc, d, use_double, use_cuaev, use_fullnbr)
