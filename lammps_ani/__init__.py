from . import utils
import torch
from torch import Tensor
from typing import Optional


class LammpsModelBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

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


__all__ = ['utils']
