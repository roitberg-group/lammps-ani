import warnings
from typing import Optional

import torch
from torch import Tensor
from torchani.nn import BmmEnsemble, Ensemble, ANINetworks
from torchani.neighbors import Neighbors
from torchani.potentials import RepulsionXTB

# disable tensorfloat32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# nvfuser and graph optimization are disabled
torch._C._jit_set_nvfuser_enabled(False)
torch._C._get_graph_executor_optimize(False)


class LammpsModelBase(torch.nn.Module):
    @torch.jit.export
    def init(self, use_cuaev: bool, use_fullnbr: bool):
        """
        Method that will be called at the very beginning within the lammps interface to set parameters.
        """
        raise NotImplementedError

    @torch.jit.export
    def forward(
        self,
        species: Tensor,
        coordinates: Tensor,
        para1: Tensor,
        para2: Tensor,
        para3: Tensor,
        species_ghost_as_padding: Tensor,
        atomic: bool = False,
        virial_flag: bool = False,
    ):
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
            virial_flag (bool, optional): Whether the virial should be returned. Defaults to False.

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


class LammpsANI(LammpsModelBase):
    def __init__(self, model, use_bmmensemble: bool = True):
        super().__init__()

        # make sure the model has correct attributes
        assert hasattr(model, "aev_computer"), "No aev_computer found in the model."
        assert hasattr(
            model, "neural_networks"
        ), "No neural_networks found in the model."
        assert isinstance(model.neural_networks, Ensemble) or isinstance(
            model.neural_networks, ANINetworks
        )
        assert hasattr(model, "energy_shifter"), "No energy_shifter found in the model."
        assert hasattr(
            model, "rep_calc"
        ), "No rep_calc is found in your model. Please set model.rep_calc = None if you don't need to calculate repulsion energy."
        assert isinstance(model.rep_calc, RepulsionXTB) or model.rep_calc is None

        # setup model
        self.use_cuaev = True
        self.use_fullnbr = True
        self.initialized = False
        self.aev_computer = model.aev_computer
        self.use_repulsion = model.rep_calc is not None
        if self.use_repulsion:
            self.rep_calc = model.rep_calc
        else:
            # We need to create a dummy repulsion model because the `compute_repulsion()`
            # method needs the repulsion forward functions are exposed.
            self.rep_calc = RepulsionXTB(symbols=("H"))

        # num_models
        self.num_models = len(model.neural_networks)
        self.use_num_models = self.num_models
        # Batched neural networks is required for selecting number of models at Runtime.
        # TODO if the normal Ensemble needs to be supported to select_models in the future,
        # A ModuleList of Ensemble with different number of models could be prepared in advance
        # within the __init__ function.
        if use_bmmensemble:
            self.neural_networks = BmmEnsemble(model.neural_networks)
        else:
            self.neural_networks = model.neural_networks
        self.energy_shifter = model.energy_shifter
        self.register_buffer("dummy_buffer", torch.empty(0))

        # we don't need weight gradient when calculating force
        for name, param in self.neural_networks.named_parameters():
            param.requires_grad_(False)

        self.using_bmmensemble = isinstance(self.neural_networks, BmmEnsemble)

    @torch.jit.export
    def init(self, use_cuaev: bool, use_fullnbr: bool):
        self.use_cuaev = use_cuaev
        self.use_fullnbr = use_fullnbr
        self.initialized = True
        if use_cuaev:
            warnings.warn("Forcing virial_flag as False because CUAEV currently does not support virial/stress/pressure calculations")

    @torch.jit.export
    def forward(
        self,
        species: Tensor,
        coordinates: Tensor,
        para1: Tensor,
        para2: Tensor,
        para3: Tensor,
        species_ghost_as_padding: Tensor,
        atomic: bool = False,
        virial_flag: bool = True,
    ):
        assert (
            self.initialized
        ), "Model is not initialized, You need to call init() method before forward function"

        if self.use_cuaev and not self.aev_computer._cuaev_computer_is_init:
            self.aev_computer._init_cuaev_computer()
            self.aev_computer._cuaev_computer_is_init = True
        # when use ghost_index and mnp, the input system must be a single molecule

        # Force virial_flag as false because cuaev does not support virial calculation
        if self.use_cuaev:
            virial_flag = False

        # prepare diff_vector for virial/stress calculations
        if self.use_fullnbr:  # although cuaev does not need this, repulsion needs to use this anyway
            # TODO: this hurt performance when repulsion is not needed and cuavev is used
            ilist_unique, jlist, numneigh = para1, para2, para3
            ilist_unique = ilist_unique.long()
            jlist = jlist.long()
            ilist = torch.repeat_interleave(ilist_unique, numneigh)
            atom_index12 = torch.cat([ilist.unsqueeze(0), jlist.unsqueeze(0)], 0)

            coords0 = coordinates.view(-1, 3).index_select(0, atom_index12[0])
            coords1 = coordinates.view(-1, 3).index_select(0, atom_index12[1])
            diff_vector = coords0 - coords1
        else:
            diff_vector = para2

        if virial_flag:
            diff_vector.requires_grad_()

        # torch.ops.mnp.nvtx_range_push("AEV forward")
        aev = self.compute_aev(species, coordinates, para1, para2, para3, fullnbr_diff_vector=diff_vector)
        # torch.ops.mnp.nvtx_range_pop()

        if atomic:
            energies, atomic_energies = self.forward_atomic(
                species, coordinates, species_ghost_as_padding, aev
            )
        else:
            energies, atomic_energies = self.forward_total(
                species, coordinates, species_ghost_as_padding, aev
            )

        if self.use_repulsion:
            # torch.ops.mnp.nvtx_range_push("Repulsion forward")
            ghost_flags = species_ghost_as_padding == -1
            rep_energies = self.compute_repulsion(
                species, coordinates, para1, para2, para3, ghost_flags, fullnbr_diff_vector=diff_vector
            )
            energies += rep_energies
            # torch.ops.mnp.nvtx_range_pop()

        if virial_flag:
            # torch.ops.mnp.nvtx_range_push("Force and Stress")
            force, dEdR = torch.autograd.grad([energies.sum()], [coordinates, diff_vector], create_graph=True, retain_graph=True)
            assert dEdR is not None
            virial = dEdR.transpose(0, 1) @ diff_vector
            virial = (virial.t() + virial) / 2
            # torch.ops.mnp.nvtx_range_pop()
        else:
            # torch.ops.mnp.nvtx_range_push("Force")
            force = torch.autograd.grad(
                [energies.sum()], [coordinates], create_graph=True, retain_graph=True
            )[0]
            # torch.ops.mnp.nvtx_range_pop()
            # When using cuaev and lammps needs to calculate pressure thermo property,
            # although we force the vflag here is False, the flag is on internally.
            # That is why these zeros are needed.
            virial = torch.zeros([3, 3])

        assert force is not None
        force = -force
        virial = -virial
        return energies, force, atomic_energies, virial

    @torch.jit.export
    def forward_total(
        self,
        species: Tensor,
        coordinates: Tensor,
        species_ghost_as_padding: Tensor,
        aev: Tensor,
    ):
        # run neural networks
        # torch.ops.mnp.nvtx_range_push(f"NN ({self.use_num_models}) forward")
        energies = self.neural_networks(species_ghost_as_padding, aev)
        # TODO force is independent of energy_shifter?
        energies += self.energy_shifter(species_ghost_as_padding)
        # torch.ops.mnp.nvtx_range_pop()

        return energies, torch.empty(0)

    @torch.jit.export
    def forward_atomic(
        self,
        species: Tensor,
        coordinates: Tensor,
        species_ghost_as_padding: Tensor,
        aev: Tensor,
    ):
        ntotal = species.shape[1]
        nghost = (species_ghost_as_padding == -1).flatten().sum()
        nlocal = ntotal - nghost

        # run neural networks
        # torch.ops.mnp.nvtx_range_push("NN ({self.use_num_models}) forward_atomic")
        atomic_energies = self.neural_networks(species_ghost_as_padding, aev, atomic=True)
        atomic_energies += self.energy_shifter(species_ghost_as_padding, atomic=True)
        # when using ANI ensemble (not batchmm), atomic_energies shape is [models, C, A]
        if len(atomic_energies.shape) > 2:
            atomic_energies = atomic_energies.mean(0)
        energies = atomic_energies.sum(dim=1)
        # torch.ops.mnp.nvtx_range_pop()

        return energies, atomic_energies[:, :nlocal]

    @torch.jit.export
    def compute_aev(
        self,
        species: Tensor,
        coordinates: Tensor,
        para1: Tensor,
        para2: Tensor,
        para3: Tensor,
        fullnbr_diff_vector: Optional[Tensor],
    ):
        atom_index12, diff_vector, distances = para1, para2, para3
        ilist_unique, jlist, numneigh = para1, para2, para3
        # compute aev
        assert (
            species.shape[0] == 1
        ), "Currently only support inference for single molecule"
        if self.use_cuaev:
            if self.use_fullnbr:
                aev = self.aev_computer._compute_cuaev_with_full_nbrlist(
                    species, coordinates, ilist_unique, jlist, numneigh
                )
            else:
                aev = self.aev_computer._cuaev_compute_from_neighbors(
                    species, coordinates, Neighbors(atom_index12, distances, diff_vector),
                )
            assert aev is not None
        else:
            if self.use_fullnbr:
                assert fullnbr_diff_vector is not None
                atom_index12, diff_vector, distances = self.aev_computer._full_to_half_nbrlist(
                    ilist_unique, jlist, numneigh, species, fullnbr_diff_vector
                )
                assert (
                    atom_index12.max() < coordinates.shape[1]
                ), f"neighbor {atom_index12.max().item()} larger than num_atoms {coordinates.shape[1]}"
            aev = self.aev_computer._pyaev_compute_from_neighbors(
                species, coordinates, Neighbors(atom_index12, distances, diff_vector),
            )

        return aev

    @torch.jit.export
    def compute_repulsion(
        self,
        species: Tensor,
        coordinates: Tensor,
        para1: Tensor,
        para2: Tensor,
        para3: Tensor,
        ghost_flags: Tensor,
        fullnbr_diff_vector: Optional[Tensor],
    ):
        atom_index12, diff_vector, distances = para1, para2, para3
        ilist_unique, jlist, numneigh = para1, para2, para3
        if self.use_fullnbr:
            assert fullnbr_diff_vector is not None
            atom_index12, diff_vector, distances = self.aev_computer._full_to_half_nbrlist(
                ilist_unique, jlist, numneigh, species, fullnbr_diff_vector
            )
            assert (
                atom_index12.max() < coordinates.shape[1]
            ), f"neighbor {atom_index12.max().item()} larger than num_atoms {coordinates.shape[1]}"
        # When using half nbrlist, distances are calculated from diff_vector, although it is before
        # we set diff_vector.requires_grad_(), the backpropogation will still work.
        # else:
        #     distances = diff_vector.norm(2, -1)
        neighbors = Neighbors(atom_index12, distances, diff_vector)
        repulsion_energies = self.rep_calc.compute_from_neighbors(
            species, coordinates, neighbors, charge=0,
            atomic=False, ensemble_values=False, ghost_flags=ghost_flags
        ).energies
        return repulsion_energies

    @torch.jit.export
    def select_models(self, use_num_models: Optional[int] = None):
        if use_num_models is None or use_num_models == self.num_models:
            pass
            # We don't need to do anything in this case, even if it is not using
            # BmmEnsemble.
        elif self.using_bmmensemble:
            raise RuntimeError("select_models method only works for BmmEnsemble")
        else:
            self.neural_networks.set_active_members(list(range(use_num_models)))
            self.use_num_models = self.neural_networks.get_active_members_num()
