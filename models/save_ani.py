import torch
import torchani


def save_cuda_aev():
    device = torch.device('cpu')
    Rcr = 5.2000e+00
    Rca = 3.5000e+00
    EtaR = torch.tensor([1.6000000e+01], device=device)
    ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00,
                         3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
    Zeta = torch.tensor([3.2000000e+01], device=device)
    ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00,
                         1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
    EtaA = torch.tensor([8.0000000e+00], device=device)
    ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00,
                         2.2000000e+00, 2.8500000e+00], device=device)
    num_species = 4
    cuaev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension=True)

    script_module = torch.jit.script(cuaev_computer)
    script_module.save('model.pt')

    # py jit test
    device = 'cuda:0'
    cu_aev = torch.jit.load('model.pt').to(device)
    coordinates = torch.tensor([
            [[0.03192167, 0.00638559, 0.01301679],
             [-0.83140486, 0.39370209, -0.26395324],
             [-0.66518241, -0.84461308, 0.20759389],
             [0.45554739, 0.54289633, 0.81170881],
             [0.66091919, -0.16799635, -0.91037834]],
            [[-4.1862600, 0.0575700, -0.0381200],
             [-3.1689400, 0.0523700, 0.0200000],
             [-4.4978600, 0.8211300, 0.5604100],
             [-4.4978700, -0.8000100, 0.4155600],
             [0.00000000, -0.00000000, -0.00000000]]
        ], device=device)
    species = torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, -1]], device=device)
    _, aev = cu_aev((species, coordinates))
    print(aev.shape)


class ANI2xHalfNbrList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=None, cell_list=True,
                                        use_cuaev_interface=True, use_cuda_extension=True, use_full_nbrlist=False)
        self.aev_computer = ani2x.aev_computer
        self.neural_networks = ani2x.neural_networks.to_infer_model(use_mnp=True)
        self.energy_shifter = ani2x.energy_shifter

    @torch.jit.export
    def forward(self, species, coordinates, atom_index12, diff_vector, distances, species_ghost_as_padding):
        if not self.aev_computer.cuaev_is_initialized:
            self.aev_computer._init_cuaev_computer()
            self.aev_computer.cuaev_is_initialized = True
            # TODO check again
            self.neural_networks.mnp_migrate_device()
        # when use ghost_index and mnp, the input system must be a single molecule
        assert species.shape[0] == 1, "Currently only support inference for single molecule"
        aev = self.aev_computer._compute_cuaev_with_nbrlist(species, coordinates, atom_index12, diff_vector, distances)
        # run neural networks
        species_energies = self.neural_networks((species_ghost_as_padding, aev))
        # TODO force is independent of energy_shifter?
        species_energies = self.energy_shifter(species_energies)
        energies = species_energies[1]
        force = torch.autograd.grad([energies.sum()], [coordinates], create_graph=True, retain_graph=True)[0]
        assert force is not None
        force = -force
        return energies, force


class ANI2xFullNbrList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=None, cell_list=True,
                                        use_cuaev_interface=True, use_cuda_extension=True, use_full_nbrlist=True)
        self.aev_computer = ani2x.aev_computer
        self.neural_networks = ani2x.neural_networks.to_infer_model(use_mnp=True)
        self.energy_shifter = ani2x.energy_shifter

    @torch.jit.export
    def forward(self, species, coordinates, ilist_unique, jlist, numneigh, species_ghost_as_padding):
        if not self.aev_computer.cuaev_is_initialized:
            self.aev_computer._init_cuaev_computer()
            self.aev_computer.cuaev_is_initialized = True
            # TODO check again
            self.neural_networks.mnp_migrate_device()
        # when use ghost_index and mnp, the input system must be a single molecule
        assert species.shape[0] == 1, "Currently only support inference for single molecule"
        aev = self.aev_computer._compute_cuaev_with_full_nbrlist(species, coordinates, ilist_unique, jlist, numneigh)
        # run neural networks
        species_energies = self.neural_networks((species_ghost_as_padding, aev))
        # TODO force is independent of energy_shifter?
        species_energies = self.energy_shifter(species_energies)
        energies = species_energies[1]
        force = torch.autograd.grad([energies.sum()], [coordinates], create_graph=True, retain_graph=True)[0]
        assert force is not None
        force = -force
        return energies, force


def save_ani2x_half_nbrlist():
    device = torch.device('cuda')
    ani2x = ANI2xHalfNbrList()
    script_module = torch.jit.script(ani2x)
    script_module.save('ani2x_cuda_half.pt')
    print("half nbrlist model saved")

    device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
    ani2x_loaded = torch.jit.load('ani2x_cuda_half.pt').to(device)
    print("loaded")
    coordinates = torch.tensor([
        [[-95.8750, -86.3210, -86.2390],
         [-95.9750, -85.5720, -85.6520],
         [-95.3300, -86.9380, -85.7510],
         [-80.5940, -82.9920, -96.5380],
         [-80.6890, -83.8700, -96.1680],
         [-81.3590, -82.8870, -97.1030],
         [-78.7080, -94.7330, -70.0690],
         [-79.4550, -95.0420, -69.5560],
         [-79.0760, -94.0700, -70.6530],
         [-93.0320, -72.7220, -95.8670],
         [-93.7370, -73.2790, -95.5370],
         [-93.0070, -71.9830, -95.2590],
         [-78.8710, -98.8470, -78.2650],
         [-79.0310, -99.6960, -78.6770],
         [-78.3610, -98.3580, -78.9110],
         [-93.2850, -81.2860, -78.5300],
         [-93.6120, -80.6310, -77.9120],
         [-92.3430, -81.1230, -78.5750],
         [-88.1110, -88.0280, -87.9190],
         [-88.7060, -88.7510, -87.7200]]], requires_grad=True, device=device)
    species = torch.tensor([[3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0]], device=device)
    ani2x = ani2x.to(device)
    atom_index12, _, diff_vector, distances = ani2x.aev_computer.neighborlist(species, coordinates)
    species_ghost_as_padding = species.detach().clone()
    # nlocal = 19
    # species_ghost_as_padding[:, nlocal:] = -1
    torch.set_printoptions(precision=10)
    energy, force = ani2x_loaded(species, coordinates, atom_index12, diff_vector, distances, species_ghost_as_padding)
    print("energy: ", energy.shape, energy.item(), energy.dtype)
    print("force : ", force.shape, force.dtype)

    energy_ref, force_ref = ani2x_ref()
    print("energy_ref: ", energy_ref.shape, energy_ref.item(), energy_ref.dtype)
    print("force_ref: ", force_ref.shape, force_ref.dtype)

    # hard-coded floating values above loss some accuracy
    threshold = 1e-4
    energy_err = torch.abs(torch.max(energy_ref.cpu() - energy.cpu()))
    force_err = torch.abs(torch.max(force_ref.cpu() - force.cpu()))
    print("energy err: ", energy_err.item())
    print("force  err: ", force_err.item())
    assert(energy_err < threshold)
    assert(force_err < threshold)
    print()


def save_ani2x_full_nbrlist():
    device = torch.device('cuda')
    ani2x = ANI2xFullNbrList()
    script_module = torch.jit.script(ani2x)
    script_module.save('ani2x_cuda_full.pt')
    print("full nbrlist model saved")

    device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
    ani2x_loaded = torch.jit.load('ani2x_cuda_full.pt').to(device)
    print("loaded")
    coordinates = torch.tensor([
        [[-95.8750, -86.3210, -86.2390],
         [-95.9750, -85.5720, -85.6520],
         [-95.3300, -86.9380, -85.7510],
         [-80.5940, -82.9920, -96.5380],
         [-80.6890, -83.8700, -96.1680],
         [-81.3590, -82.8870, -97.1030],
         [-78.7080, -94.7330, -70.0690],
         [-79.4550, -95.0420, -69.5560],
         [-79.0760, -94.0700, -70.6530],
         [-93.0320, -72.7220, -95.8670],
         [-93.7370, -73.2790, -95.5370],
         [-93.0070, -71.9830, -95.2590],
         [-78.8710, -98.8470, -78.2650],
         [-79.0310, -99.6960, -78.6770],
         [-78.3610, -98.3580, -78.9110],
         [-93.2850, -81.2860, -78.5300],
         [-93.6120, -80.6310, -77.9120],
         [-92.3430, -81.1230, -78.5750],
         [-88.1110, -88.0280, -87.9190],
         [-88.7060, -88.7510, -87.7200]]], requires_grad=True, device=device)
    species = torch.tensor([[3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0]], device=device)
    ani2x = ani2x.to(device)
    atom_index12, _, diff_vector, distances = ani2x.aev_computer.neighborlist(species, coordinates)
    ilist_unique, jlist, numneigh = ani2x.aev_computer._half_to_full_nbrlist(atom_index12)

    species_ghost_as_padding = species.detach().clone()
    # nlocal = 19
    # species_ghost_as_padding[:, nlocal:] = -1
    torch.set_printoptions(precision=10)
    energy, force = ani2x_loaded(species, coordinates, ilist_unique, jlist, numneigh, species_ghost_as_padding)
    print("energy: ", energy.shape, energy.item(), energy.dtype)
    print("force : ", force.shape, force.dtype)

    energy_ref, force_ref = ani2x_ref()
    print("energy_ref: ", energy_ref.shape, energy_ref.item(), energy_ref.dtype)
    print("force_ref: ", force_ref.shape, force_ref.dtype)

    # hard-coded floating values above loss some accuracy
    threshold = 1e-4
    energy_err = torch.abs(torch.max(energy_ref.cpu() - energy.cpu()))
    force_err = torch.abs(torch.max(force_ref.cpu() - force.cpu()))
    print("energy err: ", energy_err.item())
    print("force  err: ", force_err.item())
    assert(energy_err < threshold)
    assert(force_err < threshold)
    print()


def ani2x_ref():
    device = torch.device('cuda')
    ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=None, cell_list=True,
                                    use_cuaev_interface=True, use_cuda_extension=True).to(device)
    coordinates = torch.tensor([
        [[-95.8750, -86.3210, -86.2390],
         [-95.9750, -85.5720, -85.6520],
         [-95.3300, -86.9380, -85.7510],
         [-80.5940, -82.9920, -96.5380],
         [-80.6890, -83.8700, -96.1680],
         [-81.3590, -82.8870, -97.1030],
         [-78.7080, -94.7330, -70.0690],
         [-79.4550, -95.0420, -69.5560],
         [-79.0760, -94.0700, -70.6530],
         [-93.0320, -72.7220, -95.8670],
         [-93.7370, -73.2790, -95.5370],
         [-93.0070, -71.9830, -95.2590],
         [-78.8710, -98.8470, -78.2650],
         [-79.0310, -99.6960, -78.6770],
         [-78.3610, -98.3580, -78.9110],
         [-93.2850, -81.2860, -78.5300],
         [-93.6120, -80.6310, -77.9120],
         [-92.3430, -81.1230, -78.5750],
         [-88.1110, -88.0280, -87.9190],
         [-88.7060, -88.7510, -87.7200]]], requires_grad=True, device=device)
    species = torch.tensor([[3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0]], device=device)
    _, energies = ani2x((species, coordinates))
    force = -torch.autograd.grad(energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]

    return energies, force


if __name__ == '__main__':
    # save_cuda_aev()
    # save_ani2x()
    save_ani2x_half_nbrlist()
    save_ani2x_full_nbrlist()
