import torch
import torchani
from ase.io import read
import argparse

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class ANI2xNoCUAEV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=None, cell_list=False,
                                        use_cuaev_interface=False, use_cuda_extension=False)
        self.aev_computer = ani2x.aev_computer
        # self.neural_networks = ani2x.neural_networks.to_infer_model(use_mnp=True)
        self.neural_networks = ani2x.neural_networks
        self.energy_shifter = ani2x.energy_shifter
        self.dummy_param = torch.nn.Parameter(torch.empty(0))


    @torch.jit.export
    def forward(self, species, coordinates, atom_index12, diff_vector, distances, species_ghost_as_padding):
        # if not self.aev_computer.cuaev_is_initialized:
            # self.aev_computer._init_cuaev_computer()
            # self.aev_computer.cuaev_is_initialized = True
            # TODO check again
            # self.neural_networks.mnp_migrate_device()
        # when use ghost_index and mnp, the input system must be a single molecule

        # convert dtype
        dtype = self.dummy_param.dtype
        ntotal = species.shape[1]
        nghost = (species_ghost_as_padding == -1).flatten().sum()
        nlocal = ntotal - nghost

        diff_vector = diff_vector.to(dtype)
        distances = distances.to(dtype)

        # compute
        assert species.shape[0] == 1, "Currently only support inference for single molecule"
        # aev = self.aev_computer._compute_cuaev_with_nbrlist(species, coordinates, atom_index12, diff_vector, distances)
        aev = self.aev_computer._compute_aev(species, atom_index12, diff_vector, distances)
        # run neural networks
        species_energies = self.neural_networks((species_ghost_as_padding, aev))
        # TODO force is independent of energy_shifter?
        species_energies = self.energy_shifter(species_energies)
        energies = species_energies[1]
        force = torch.autograd.grad([energies.sum()], [coordinates], create_graph=True, retain_graph=True)[0]
        assert force is not None
        force = -force
        return energies, force



def save_ani2x_model(runpbc=False):
    device = torch.device('cuda')
    hartree2kcalmol = 627.5094738898777

    # dtype
    use_double = True
    dtype = torch.float64 if use_double else torch.float32

    ani2x = ANI2xNoCUAEV()
    ani2x = ani2x.to(dtype)
    script_module = torch.jit.script(ani2x)
    filename = 'ani2x_cuda_nocuaev_double.pt'
    script_module.save(filename)

    device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
    ani2x_loaded = torch.jit.load(filename).to(device)
    ani2x_ref = torchani.models.ANI2x(periodic_table_index=False, model_index=None, cell_list=False,
                                      use_cuaev_interface=False, use_cuda_extension=False).to(device)
    ani2x_ref = ani2x_ref.to(dtype)
    input_file = "water-0.8nm.pdb"
    mol = read(input_file)

    species = torch.tensor(mol.get_atomic_numbers(), device=device).unsqueeze(0)
    coordinates = torch.tensor(mol.get_positions(), dtype=dtype, requires_grad=True, device=device).unsqueeze(0)
    species, coordinates = ani2x_ref.species_converter((species, coordinates))
    cell = torch.tensor(mol.cell, device=device, dtype=dtype)
    pbc = torch.tensor(mol.pbc, device=device)

    # TODO It is IMPORTANT to set cutoff as 7.1 to match lammps nbr cutoff
    ani2x_ref.aev_computer.neighborlist.cutoff = 7.1
    if runpbc:
        atom_index12, _, diff_vector, distances = ani2x_ref.aev_computer.neighborlist(species, coordinates, cell, pbc)
    else:
        atom_index12, _, diff_vector, distances = ani2x_ref.aev_computer.neighborlist(species, coordinates)
    print(distances.shape)
    species_ghost_as_padding = species.detach().clone()
    torch.set_printoptions(profile="full")

    torch.set_printoptions(precision=13)
    energy, force = ani2x_loaded(species, coordinates, atom_index12, diff_vector, distances, species_ghost_as_padding)
    print("energy: ", energy.shape, energy.item(), energy.dtype)
    energy, force = energy * hartree2kcalmol, force * hartree2kcalmol
    print("energy: ", energy.shape, energy.item(), energy.dtype)
    print("force : ", force.shape, force.dtype)

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
    print("energy_ref: ", energy_ref.shape, energy_ref.item(), energy_ref.dtype)
    print("force_ref: ", force_ref.shape, force_ref.dtype)

    threshold = 1e-7
    energy_err = torch.abs(torch.max(energy_ref.cpu() - energy.cpu()))
    force_err = torch.abs(torch.max(force_ref.cpu() - force.cpu()))

    print("energy err: ", energy_err.item())
    print("force  err: ", force_err.item())
    assert(energy_err < threshold)
    assert(force_err < threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pbc', default=False, action='store_true')
    args = parser.parse_args()
    save_ani2x_model(args.pbc)
