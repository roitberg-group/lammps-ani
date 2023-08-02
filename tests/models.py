import torch
import torchani
import warnings
from torchani.nn import ANIModel
from torchani.models import Ensemble
from .lammps_ani import LammpsANI
from torchani.potentials.repulsion import RepulsionXTB
from ani2x_ext.custom_emsemble_ani2x_ext import CustomEnsemble


def ANI2x_Model():
    model = torchani.models.ANI2x(
        periodic_table_index=True,
        model_index=None,
        cell_list=False,
        use_cuaev_interface=True,
        use_cuda_extension=True,
    )
    model.rep_calc = None
    return model


def ANI1x_NR_Model(use_repulsion):
    from torchani.models import BuiltinModel, _load_ani_model
    from pathlib import Path

    def ANI1x_NR(**kwargs) -> BuiltinModel:
        """
        Machine learning interatomic potential for condensed-phase reactive chemistry

        From: https://github.com/atomistic-ml/ani-1xnr

        Reference:
        ZHANG, S.; Makoś, M.; Jadrich, R.; Kraka, E.; Barros, K.; Nebgen, B.; Tretiak,
        S.; Isayev, O.; Lubbers, N.; Messerly, R.; Smith, J. Exploring the Frontiers
        of Chemistry with a General Reactive Machine Learning Potential. 2022.
        https://doi.org/10.26434/chemrxiv-2022-15ct6-v2.
        """
        info_file = Path('../external/ani-1xnr/model/ani-1xnr.info').absolute()
        state_dict_file = None
        return _load_ani_model(state_dict_file, info_file, use_neurochem_source=True, **kwargs)

    model = ANI1x_NR(
        periodic_table_index=True,
        model_index=None,
        cell_list=False,
        use_cuaev_interface=True,
        use_cuda_extension=True,
        # [TODO] we would need to set repulsion if we need to run ANI1x_NR with ASE calculator
        # pretrained=False,
        # repulsion=use_repulsion,
        # # The repulsion cutoff is set to 5.1, but ANIdr model actually uses a cutoff of 5.3
        # repulsion_kwargs={
        #     "symbols": ("H", "C", "N", "O"),
        #     "cutoff": 5.1,
        #     "cutoff_fn": "smooth2",
        # },
    )

    # The ANI1x_NR model does not have repulsion calculator, so the repulsion calculator here is
    # an external potential added on top of the ANI1x_NR model to prevent atoms from collapsing.
    if use_repulsion:
        model.rep_calc = RepulsionXTB(cutoff=5.1, symbols=("H", "C", "N", "O"), cutoff_fn="smooth2")
    else:
        model.rep_calc = None
    return model


def ANI2x_Repulsion_Model():
    elements = ("H", "C", "N", "O", "S", "F", "Cl")

    def dispersion_atomics(atom: str = "H"):
        dims_for_atoms = {
            "H": (1008, 256, 192, 160),
            "C": (1008, 256, 192, 160),
            "N": (1008, 192, 160, 128),
            "O": (1008, 192, 160, 128),
            "S": (1008, 160, 128, 96),
            "F": (1008, 160, 128, 96),
            "Cl": (1008, 160, 128, 96),
        }
        return torchani.atomics.standard(
            dims_for_atoms[atom], activation=torch.nn.GELU(), bias=False
        )

    model = torchani.models.ANI2x(
        pretrained=False,
        cutoff_fn="smooth",
        atomic_maker=dispersion_atomics,
        ensemble_size=7,
        repulsion=True,
        # The repulsion cutoff is set to 5.1, but ANIdr model actually uses a cutoff of 5.3
        repulsion_kwargs={
            "symbols": elements,
            "cutoff": 5.1,
            "cutoff_fn": torchani.cutoffs.CutoffSmooth(order=2),
        },
        periodic_table_index=True,
        model_index=None,
        cell_list=False,
        use_cuaev_interface=True,
        use_cuda_extension=True,
    )
    state_dict = torchani.models._fetch_state_dict(
        "anid_state_dict_mod.pt", private=True
    )
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
    # setup repulsion calculator
    model.rep_calc = model.potentials[0]

    return model


class ANI2xExt_Model(CustomEnsemble):
    """
    ani_ext model with repulsion, smooth cutoff, GELU, No Bias, GSAE
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aev_computer = torchani.AEVComputer.like_2x(
            cutoff_fn="smooth", use_cuda_extension=True, use_cuaev_interface=True
        )
        self.neural_networks = self.models
        self.species_converter = self.number2tensor
        self.rep_calc = RepulsionXTB(
            cutoff=5.1, symbols=("H", "C", "N", "O", "S", "F", "Cl")
        )

    def forward(self):
        raise RuntimeError("forward is not suppported")


all_models_ = {
    "ani2x.pt": {"model": ANI2x_Model, "unittest": True},
    "ani2x_repulsion.pt": {"model": ANI2x_Repulsion_Model, "unittest": True},
    "ani1x_nr.pt": {"model": ANI1x_NR_Model, "unittest": True, "kwargs": {"use_repulsion": False}},
    "ani1x_nr_repulsion.pt": {"model": ANI1x_NR_Model, "unittest": False, "kwargs": {"use_repulsion": True}},
    # Because ani2x_ext uses public torchani that has legacy aev code, we cannot run unittest for it.
    "ani2x_ext0_repulsion.pt": {"model": ANI2xExt_Model, "unittest": False, "kwargs": {"model_choice": 0}},
    "ani2x_ext2_repulsion.pt": {"model": ANI2xExt_Model, "unittest": False, "kwargs": {"model_choice": 2}},
}
all_models = {}

# Remove model that cannot be instantiated, e.g. ani2x_repulsion could only be downloaded within UF network
for output_file, info in all_models_.items():
    try:
        if "kwargs" in info:
            kwargs = info["kwargs"]
        else:
            kwargs = {}
        model = info["model"](**kwargs)
        all_models[output_file] = info
    except Exception as e:
        warnings.warn(f"Failed to export {output_file}: {str(e)}")

def save_models():
    for output_file, info in all_models.items():
        print(f"saving model: {output_file}")
        if "kwargs" in info:
            kwargs = info["kwargs"]
        else:
            kwargs = {}
        model = info["model"](**kwargs)
        ani2x = LammpsANI(model)
        script_module = torch.jit.script(ani2x)
        script_module.save(output_file)
