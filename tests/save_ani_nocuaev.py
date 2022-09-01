import torch
import torchani
from ase.io import read
import argparse
from typing import Tuple, Optional
from torch import Tensor
from torchani.nn import SpeciesEnergies

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class ANI2x(torch.nn.Module):
    def __init__(self, use_cuaev):
        super().__init__()
        ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=None, cell_list=False,
                                      use_cuaev_interface=use_cuaev, use_cuda_extension=use_cuaev)
        self.use_cuaev = use_cuaev
        self.aev_computer = ani2x.aev_computer
        # self.neural_networks = ani2x.neural_networks.to_infer_model(use_mnp=True)
        self.neural_networks = ani2x.neural_networks
        self.energy_shifter = ani2x.energy_shifter
        self.register_buffer("dummy_buffer", torch.empty(0))

    @torch.jit.export
    def forward(self, species, coordinates, atom_index12, diff_vector, distances, species_ghost_as_padding, atomic: bool=False):
        if self.use_cuaev and not self.aev_computer.cuaev_is_initialized:
            self.aev_computer._init_cuaev_computer()
            self.aev_computer.cuaev_is_initialized = True
            # TODO check again
            # self.neural_networks.mnp_migrate_device()
        # when use ghost_index and mnp, the input system must be a single molecule

        # dtype
        dtype = self.dummy_buffer.dtype

        # compute aev
        assert species.shape[0] == 1, "Currently only support inference for single molecule"
        if self.use_cuaev:
            # diff_vector, distances, coordinates from lammps are always in double,
            # cuaev currently only works with single precision
            diff_vector = diff_vector.to(torch.float32)
            distances = distances.to(torch.float32)
            coordinates = coordinates.to(torch.float32)
            aev = self.aev_computer._compute_cuaev_with_nbrlist(species, coordinates, atom_index12, diff_vector, distances)
            # the neural network part will use whatever dtype the user specified
            aev = aev.to(dtype)
        else:
            # diff_vector, distances from lammps are always in double,
            # we need to convert it to single precision if needed
            diff_vector = diff_vector.to(dtype)
            distances = distances.to(dtype)
            aev = self.aev_computer._compute_aev(species, atom_index12, diff_vector, distances)

        # run neural networks
        if atomic:
            return self.forward_atomic(coordinates, species_ghost_as_padding, aev)
        else:
            return self.forward_total(coordinates, species_ghost_as_padding, aev)

    @torch.jit.export
    def forward_total(self, coordinates, species_ghost_as_padding, aev):
        # run neural networks
        species_energies = self.neural_networks((species_ghost_as_padding, aev))
        # TODO force is independent of energy_shifter?
        species_energies = self.energy_shifter(species_energies)
        energies = species_energies[1]
        force = torch.autograd.grad([energies.sum()], [coordinates], create_graph=True, retain_graph=True)[0]
        assert force is not None
        force = -force
        return energies, force, torch.empty(0)

    @torch.jit.export
    def forward_atomic(self, coordinates, species_ghost_as_padding, aev):
        ntotal = species_ghost_as_padding.shape[1]
        nghost = (species_ghost_as_padding == -1).flatten().sum()
        nlocal = ntotal - nghost

        # run neural networks
        atomic_energies = self.neural_networks._atomic_energies((species_ghost_as_padding, aev))
        atomic_energies += self.energy_shifter._atomic_saes(species_ghost_as_padding)
        if atomic_energies.dim() == 2:
            atomic_energies = atomic_energies.unsqueeze(0)
        atomic_energies = atomic_energies.mean(dim=0)

        energies = atomic_energies.sum(dim=1)
        force = torch.autograd.grad([energies.sum()], [coordinates], create_graph=True, retain_graph=True)[0]
        assert force is not None
        force = -force
        return energies, force, atomic_energies[:, :nlocal]

class ANI2xRef(torch.nn.Module):
    """
    This is used to handel cuaev_computer that currently only works with single precision.
    """
    def __init__(self, use_cuaev):
        super().__init__()
        ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=None, cell_list=False,
                                      use_cuaev_interface=use_cuaev, use_cuda_extension=use_cuaev)
        self.model = ani2x
        self.use_cuaev = use_cuaev
        self.register_buffer("dummy_buffer", torch.empty(0))

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


def save_ani2x_model(runpbc=False, device='cuda', use_double=True, use_cuaev=False):
    hartree2kcalmol = 627.5094738898777

    # dtype
    dtype = torch.float64 if use_double else torch.float32

    ani2x = ANI2x(use_cuaev)
    ani2x = ani2x.to(dtype)
    # cuaev currently only works with single precision
    if use_cuaev:
        ani2x.aev_computer = ani2x.aev_computer.to(torch.float32)
    script_module = torch.jit.script(ani2x)
    script_module.save(output_file)

    ani2x_loaded = torch.jit.load(output_file).to(device)
    # ani2x_loaded = ani2x.to(device)
    ani2x_ref = ANI2xRef(use_cuaev).to(device).to(dtype)
    # cuaev currently only works with single precision
    if use_cuaev:
        ani2x_ref.model.aev_computer = ani2x_ref.model.aev_computer.to(torch.float32)
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
    print(distances.shape)
    species_ghost_as_padding = species.detach().clone()
    torch.set_printoptions(profile="full")

    torch.set_printoptions(precision=13)
    energy, force, _ = ani2x_loaded(species, coordinates, atom_index12, diff_vector, distances, species_ghost_as_padding, atomic=False)

    # test forward_atomic API
    energy_, force_, atomic_energies = ani2x_loaded(species, coordinates, atom_index12, diff_vector, distances, species_ghost_as_padding, atomic=True)
    assert torch.allclose(energy, energy_)
    assert torch.allclose(force, force_)
    assert torch.allclose(energy, atomic_energies.sum(dim=-1))

    energy, force = energy * hartree2kcalmol, force * hartree2kcalmol
    print(f"{'energy:'.ljust(15)} shape: {energy.shape}, value: {energy.item()}, dtype: {energy.dtype}, unit: (kcal/mol)")
    print(f"{'force:'.ljust(15)} shape: {force.shape}, dtype: {force.dtype}, unit: (kcal/mol/A)")

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
    print(f"{'energy_ref:'.ljust(15)} shape: {energy_ref.shape}, value: {energy_ref.item()}, dtype: {energy_ref.dtype}, unit: (kcal/mol)")
    print(f"{'force_ref:'.ljust(15)} shape: {force_ref.shape}, dtype: {force_ref.dtype}, unit: (kcal/mol/A)")

    threshold = 1e-7 if use_double else 3e-5
    energy_err = torch.abs(torch.max(energy_ref.cpu() - energy.cpu()))
    force_err = torch.abs(torch.max(force_ref.cpu() - force.cpu()))

    print("energy max err: ".ljust(15), energy_err.item())
    print("force  max err: ".ljust(15), force_err.item())
    assert(energy_err < threshold)
    assert(force_err < threshold)


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
        cuaev_or_nocuaev = "cuaev" if use_cuaev else "nocuaev"
        for use_double in [True, False]:
            double_or_single = "double" if use_double else "single"
            output_file = f'ani2x_{cuaev_or_nocuaev}_{double_or_single}.pt'
            print(output_file)
            for pbc in [False, True]:
                for d in devices:
                    if use_cuaev and d == "cpu":
                        continue
                    print(f"====================== {cuaev_or_nocuaev} | {double_or_single} | pbc: {pbc} | device: {d} ======================")
                    save_ani2x_model(pbc, d, use_double, use_cuaev)
