import subprocess
import os
import datetime
from typing import Dict

LAMMPS_PATH = os.path.join(os.environ["LAMMPS_ROOT"], "build/lmp_mpi")


class LammpsRunner:
    def __init__(
        self,
        lmp: str,
        input_file: str,
        var_dict: Dict,
        kokkos: bool,
        num_gpus: int = 1,
        log_dir: str = "logs",
        run_name: str = "run",
        allow_tf32: bool = False,
    ):
        # additional variables
        var_dict["newton_pair"] = "on" if kokkos else "off"
        var_dict["ani_device"] = "cuda"
        var_dict["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

        # create run commands
        var_commands = " ".join(
            [f"-var {var} {value}" for var, value in var_dict.items()]
        )
        kokkos_commands = (
            f"-k on g {num_gpus} -sf kk -pk kokkos gpu/aware on" if kokkos else ""
        )

        # Create logs directory and logfile name
        os.makedirs(log_dir, exist_ok=True)
        logfile = f"{log_dir}/{var_dict['timestamp']}-{'kokkos-' if kokkos == 'yes' else ''}models_{var_dict['ani_num_models']}-gpus_{num_gpus}-{run_name}.log"
        env_vars = ""
        if kokkos:
            env_vars += "LAMMPS_ANI_PROFILING=1 "
        if allow_tf32:
            env_vars += "LAMMPS_ANI_ALLOW_TF32=1 "

        run_commands = (
            f"{env_vars} mpirun -np {num_gpus} {lmp} "
            f"{var_commands} {kokkos_commands} "
            f"-in {input_file} -log {logfile}"
        )
        print(f"Run with command:\n{run_commands}")
        self.run_commands = run_commands

    def run(self):
        stdout = subprocess.run(
            self.run_commands,
            shell=True,
            check=True,
        )


# run parameters
run_name = "run_nvt"
num_gpus = 1
kokkos = True
log_dir = "logs"
input_file = "in.lammps"
allow_tf32 = False

# variables for lammps input file
var_dict = {
    # configuration
    "data_file": "data/water-150k.data",
    "timestep": 0.5,
    "run_steps": 1000,
    # ani variables
    "ani_model_file": os.getenv("LAMMPS_ANI_ROOT") + "/tests/ani2x.pt",
    "ani_num_models": -1,  # -1 means use all models
    "ani_aev": "cuaev",
    "ani_neighbor": "full",
    "ani_precision": "single",
}

LammpsRunner(
    LAMMPS_PATH, input_file, var_dict, kokkos, num_gpus, log_dir, run_name, allow_tf32
)
