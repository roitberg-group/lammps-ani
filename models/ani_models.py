import warnings
from pathlib import Path

import torch
from torchani.potentials import RepulsionXTB
from torchani.models import ANI2x
from torchani.neurochem import load_model_from_info_file

from .lammps_ani import LammpsANI

# Get the directory where this module is located
_MODULE_DIR = Path(__file__).parent.absolute()
_EXTERNAL_DIR = _MODULE_DIR.parent / "external"


def ANI2x_Model():
    model = ANI2x(neighborlist="all_pairs", strategy="cuaev")
    model.rep_calc = None
    return model


def ANI1x_NR_Model(use_repulsion):

    def ANI1x_NR(**kwargs):
        """
        Machine learning interatomic potential for condensed-phase reactive chemistry

        From: https://github.com/atomistic-ml/ani-1xnr

        Reference:
        ZHANG, S.; Makoś, M.; Jadrich, R.; Kraka, E.; Barros, K.; Nebgen, B.; Tretiak,
        S.; Isayev, O.; Lubbers, N.; Messerly, R.; Smith, J. Exploring the Frontiers
        of Chemistry with a General Reactive Machine Learning Potential. 2022.
        https://doi.org/10.26434/chemrxiv-2022-15ct6-v2.
        """
        # Create symlink in expected location if it doesn't exist
        # torchani looks for files in neurochem_dir()/ani-1xnr/
        from torchani.paths import neurochem_dir
        target_dir = neurochem_dir() / "ani-1xnr"
        source_dir = _EXTERNAL_DIR / "ani-1xnr" / "model" / "ani-1xnr"
        if not target_dir.exists():
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            target_dir.symlink_to(source_dir)

        info_file = _EXTERNAL_DIR / "ani-1xnr" / "model" / "ani-1xnr.info"
        return load_model_from_info_file(info_file, **kwargs)

    model = ANI1x_NR(strategy="cuaev")

    # The ANI1x_NR model does not have repulsion calculator, so the repulsion calculator here is
    # an external potential added on top of the ANI1x_NR model to prevent atoms from collapsing.
    if use_repulsion:
        model.rep_calc = RepulsionXTB(cutoff=5.1, symbols=("H", "C", "N", "O"), cutoff_fn="smooth")
    else:
        model.rep_calc = None
    return model


def ANI2x_Solvated_Alanine_Dipeptide_Model():
    try:
        import ani_engine.utils
    except ImportError:
        raise RuntimeError("ani_engine is not installed, cannot export ANI2x_Solvated_Alanine_Dipeptide_Model")
    engine = ani_engine.utils.load_engine(str(_EXTERNAL_DIR / "ani_engine_models/20230913_131808-zdy6gco1-2x-with-solvated-alanine-dipeptide-b973c-def2-mtzvp"))
    # Use cuaev if CUDA available, otherwise fall back to CPU-compatible export
    use_cuaev = torch.cuda.is_available()
    model = engine.model.to_builtins(engine.self_energies, use_cuaev_interface=use_cuaev)
    model.rep_calc = None
    return model


def ANI2x_B973c():
    """
    ANI2x model with B973c dataset, no new solvated alanine dipeptide data
    """
    try:
        import ani_engine.utils
    except ImportError:
        raise RuntimeError("ani_engine is not installed, cannot export ANI2x_B973c")
    engine = ani_engine.utils.load_engine(str(_EXTERNAL_DIR / "ani_engine_models/20230906_120322-7avzat0g-2x-energy-force-b973c-no_new_data"))
    # Use cuaev if CUDA available, otherwise fall back to CPU-compatible export
    use_cuaev = torch.cuda.is_available()
    model = engine.model.to_builtins(engine.self_energies, use_cuaev_interface=use_cuaev)
    model.rep_calc = None
    return model


all_models_ = {
    "ani2x.pt": {"model": ANI2x_Model, "unittest": True},
    "ani1x_nr.pt": {"model": ANI1x_NR_Model, "unittest": True, "kwargs": {"use_repulsion": False}},
    # trained with energy+force, B973c/def2-mTZVP level of theory
    # - ani2x_solvated_alanine_dipeptide.pt: WITH solvated alanine dipeptide data
    # "ani2x_solvated_alanine_dipeptide.pt": {"model": ANI2x_Solvated_Alanine_Dipeptide_Model, "unittest": True},
    # - ani2x_b973c.pt: WITHOUT solvated alanine dipeptide data
    # "ani2x_b973c.pt": {"model": ANI2x_B973c, "unittest": True},
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
        print(f"Failed to export {output_file}: {str(e)}")


def save_models():
    for output_file, info in all_models.items():
        print(f"saving model: {output_file}")
        if "kwargs" in info:
            kwargs = info["kwargs"]
        else:
            kwargs = {}
        model = info["model"](**kwargs)
        m = LammpsANI(model)
        script_module = torch.jit.script(m)
        script_module.save(output_file)
