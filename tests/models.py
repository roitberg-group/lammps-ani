import torch
import torchani
from torchani.nn import ANIModel
from torchani.models import Ensemble
from .lammps_ani import LammpsANI
from ani2x_ext.custom_emsemble_ani2x_ext import CustomEnsemble


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


all_models = {"ani2x.pt": {"model": ANI2x_Model, "use_repulsion": False},
              "ani2x_repulsion.pt": {"model": ANI2x_Repulsion_Model, "use_repulsion": True},
            #   "ani1x_zeng.pt": {"model": ANI1x_Zeng, "use_repulsion": True},
            #   "ani2x_ext0_repulsion": {"model": ANI2xExt_Model, "use_repulsion": True},
              }


# TODO rename this function name
def save_models():
    for output_file, info in all_models.items():
        ani2x = LammpsANI(info["model"](), use_repulsion=info["use_repulsion"])
        script_module = torch.jit.script(ani2x)
        script_module.save(output_file)

