import torch
import torchani
from ase.io import read

import torchsnooper
import snoop

torchsnooper.register_snoop(verbose=True)
torch.set_printoptions(precision=15)


#@snoop
def func():
    input_file = "water-0.8nm.pdb"
    mol = read(input_file)

    device = torch.device("cuda")
    ani2x = torchani.models.ANI2x(
        periodic_table_index=False,
        model_index=None,
        cell_list=True,
        use_cuaev_interface=True,
        use_cuda_extension=True,
    ).to(device)

    species = torch.tensor([mol.get_atomic_numbers()], device=device)
    coordinates = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=True, device=device)
    species, coordinates = ani2x.species_converter((species, coordinates))
    cell = torch.tensor(mol.cell, device=device, dtype=torch.float32)
    pbc = torch.tensor(mol.pbc, device=device)

    # _, energies = ani2x((species, coordinates), cell, pbc)
    _, energies = ani2x((species, coordinates))
    force = -torch.autograd.grad(
        energies.sum(), coordinates, create_graph=True, retain_graph=True
    )[0]

    hartree2kcalmol = 627.5094738898777
    print(coordinates)
    print(energies * hartree2kcalmol)
    print(force * hartree2kcalmol)


func()
