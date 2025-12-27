import os
import pprint
import datetime
import argparse
import subprocess
from typing import Dict
from pathlib import Path

LAMMPS_PATH = os.path.join(os.environ["LAMMPS_ROOT"], "build/lmp_mpi")


class LammpsRunner:
    def __init__(
        self,
        lmp: str,
        input_file: str,
        var_dict: Dict,
        kokkos: bool,
        num_gpus: int = 1,
        run_name: str = "run",
        allow_tf32: bool = False,
        nsys_profile: bool = False,
    ):
        """
        Initialize LammpsRunner.
        """
        var_dict["newton_pair"] = "on" if kokkos else "off"
        var_dict["ani_device"] = "cuda"
        var_dict["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S.%f")

        # create run commands
        var_commands = " ".join(
            [f"-var {var} {value}" for var, value in var_dict.items()]
        )
        kokkos_commands = (
            f"-k on g {num_gpus} -sf kk -pk kokkos gpu/aware on" if kokkos else ""
        )

        # Create logs directory and logfile name
        os.makedirs(var_dict["log_dir"], exist_ok=True)
        data_filename = Path(var_dict["data_file"]).stem
        logfile = os.path.join(
            var_dict["log_dir"],
            f"{var_dict['timestamp']}-label_{var_dict['label']}-{'kokkos-' if kokkos else ''}models_{var_dict['ani_num_models']}-gpus_{num_gpus}-{data_filename}-{run_name}.log",
        )

        self.run_commands = (
            f"mpirun -np {num_gpus} {lmp} "
            f"{var_commands} {kokkos_commands} "
            f"-in {input_file} -log {logfile}"
        )

        add_env_vars = {}  # additional environment variables
        if kokkos:
            add_env_vars["LAMMPS_ANI_PROFILING"] = "1"
        if allow_tf32:
            add_env_vars["LAMMPS_ANI_ALLOW_TF32"] = "1"

        env_vars = ""
        for k, v in add_env_vars.items():
            env_vars += f"{k}={v} "

        self.var_dict = var_dict
        if nsys_profile:
            nsys_commands = f"nsys profile --stats=true --trace cuda,nvtx,mpi --cudabacktrace=true -o {var_dict['log_dir'] + '/' +var_dict['timestamp']}.qdrep "
            self.run_commands = env_vars + nsys_commands + self.run_commands
        else:
            self.run_commands = env_vars + self.run_commands
        print(f"Run with command:\n{self.run_commands}", flush=True)

    def run(self):
        """
        Run the LAMMPS simulation.
        """
        try:
            subprocess.run(
                self.run_commands,
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running LAMMPS: {e}")

    def write_conf(self, args):
        formatted_dict = pprint.pformat(args.__dict__)
        with open(f"{self.var_dict['log_dir']}/{self.var_dict['timestamp']}.conf", 'w') as file:
            file.write(formatted_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LAMMPS simulation.")
    parser.add_argument("data_file", help="Data file for the simulation.")

    parser.add_argument("--input_file", default="in.lammps", help="LAMMPS input file.")
    parser.add_argument("--run_name", default="run", help="Name of the run.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--kokkos", action="store_true", help="Use Kokkos.")
    parser.add_argument("--log_dir", default="logs", help="Directory to store log files.")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TensorFlow 32.")

    parser.add_argument("--label", type=str, help="Label for the umbrella window.")
    parser.add_argument("--timestep", type=float, default=0.5, help="Timestep for the simulation.")
    parser.add_argument("--run_steps", type=int, default=1000, help="Number of steps for the simulation.")
    parser.add_argument("--replicate", type=str, default="1 1 1", help="Replicate the current simulation one or more times in each dimension")
    parser.add_argument("--ani_model_file", default="ani2x.pt", help="ANI model file.")
    parser.add_argument("--ani_num_models", type=int, default=1, help="Number of ANI models to use. -1 means use all models.")
    parser.add_argument("--ani_aev", choices=["cuaev", "pyaev"], default="cuaev", help="ANI AEV method.")
    parser.add_argument("--ani_neighbor", choices=["full", "half"], default="full", help="ANI neighbor list.")
    parser.add_argument("--ani_precision", choices=["single", "double"], default="single", help="ANI precision method.")

    parser.add_argument("--run", action="store_true", help="If specified, run the simulation.")
    parser.add_argument("--nsys", action="store_true", help="If specified, run nsight profiling.")

    args = parser.parse_args()
    args.ani_model_file = os.path.join(os.getenv("LAMMPS_ANI_ROOT"), "models", args.ani_model_file)

    var_dict = {
        # configuration
        "data_file": args.data_file,
        "timestep": args.timestep,
        "run_steps": args.run_steps,
        "log_dir": args.log_dir,
        "label": args.label,
        "replicate": f"'{args.replicate}'",
        # ani variables
        "ani_model_file": args.ani_model_file,
        "ani_num_models": args.ani_num_models,
        "ani_aev": args.ani_aev,
        "ani_neighbor": args.ani_neighbor,
        "ani_precision": args.ani_precision,
    }

    # Pretty print all arguments
    pprint.pprint(args.__dict__)

    lmp_runner = LammpsRunner(
        LAMMPS_PATH, args.input_file, var_dict, args.kokkos, args.num_gpus, args.run_name, args.allow_tf32, args.nsys
    )

    # Only run if --run is specified
    if args.run:
        # save the configurations
        lmp_runner.write_conf(args)
        # run
        lmp_runner.run()
    else:
        print("Simulation not run. Use --run to start the simulation.")
