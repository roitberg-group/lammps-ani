import time
import argparse
from simple_slurm import Slurm


def calculate_nodes_and_tasks(num_gpus, weak_scaling=False):
    # Logic to calculate nodes and tasks per node based on number of GPUs
    if num_gpus <= 8:
        nodes = 1
        ntasks_per_node = num_gpus
        gres = f"gpu:{num_gpus}"
        if num_gpus == 1:
            replicate = "1 1 1"
        elif num_gpus == 2:
            replicate = "2 1 1"
        elif num_gpus == 4:
            replicate = "2 2 1"
        elif num_gpus == 8:
            replicate = "2 2 2"
        else:
            raise ValueError("Number of GPUs should be 1, 2, 4 or 8")
    else:
        assert (
            num_gpus % 8 == 0
        ), "Number of GPUs should be a multiple of 8 for more than 8 GPUs"
        nodes = num_gpus // 8
        ntasks_per_node = 8
        gres = f"gpu:{8}"
        replicate = f"2 2 {nodes * 2}"
    if not weak_scaling:
        replicate = "1 1 1"
    return nodes, ntasks_per_node, gres, replicate


def setup_and_run_job(num_gpus, data_file, job_name, submit=False, weak_scaling=False, log_dir="logs"):
    # Variables for easy adjustment
    output_filename = f"{job_name}_%j_{num_gpus}GPUs.log"

    nodes, ntasks_per_node, gres, replicate = calculate_nodes_and_tasks(num_gpus, weak_scaling)

    slurm = Slurm(
        job_name=job_name,
        ntasks=num_gpus,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        cpus_per_task=1,
        partition="hpg-ai",
        reservation="roitberg",
        qos="roitberg",
        account="roitberg",
        gres=gres,
        mem_per_cpu="100gb",
        time="120:00:00",
        output=output_filename,
    )

    commands = [
        'echo "Date              = $(date)"',
        'echo "Hostname          = $(hostname -s)"',
        'echo "Working Directory = $(pwd)"',
        'echo ""',
        'echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"',
        'echo "Nodes Allocated                = $SLURM_JOB_NODELIST"',
        'echo "Number of Tasks Allocated      = $SLURM_NTASKS"',
        'echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"',
        # module load and setup environment variables
        "module load cuda/11.4.3 gcc/9.3.0 openmpi/4.0.5 cmake/3.21.3 git/2.30.1",
        'export LAMMPS_ANI_ROOT="/blue/roitberg/apps/lammps-ani"',
        "export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/",
        "export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/",
        # setup conda in the subshell and activate the environment
        # check issue: https://github.com/conda/conda/issues/7980
        "source $(conda info --base)/etc/profile.d/conda.sh",
        "conda activate /blue/roitberg/apps/torch1121",
        "echo using python: $(which python)",
        # run the job commands
        # "python run_one.py --help",
        # --allow_tf32
        f"python run_one.py {data_file} --input_file=in.lammps --replicate='{replicate}' --kokkos --num_gpus={num_gpus} --run_steps=5000 --run_name='run' --log_dir={log_dir} --run"
    ]
    commands = "\n".join(commands)
    if submit:
        slurm.sbatch(commands, convert=False)
        # prevent submitting too fast that results in the same timestamp
        time.sleep(1)
    else:
        print("Job will not be submitted. The following is the job script:")
        print(str(slurm) + commands)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LAMMPS-ANI benchmarks.")
    # parser.add_argument("num_gpus", type=int, help="Number of GPUs to use for the job.")
    parser.add_argument("data_file", type=str, help="Path to the data file.")
    parser.add_argument('-y', action='store_true', help='If provided, the job will be submitted. If not, the job will only be prepared but not submitted.')
    args = parser.parse_args()

    # control weak or strong scaling
    weak_scaling = False
    log_dir = "log_water_strong_scaling"
    # run
    if weak_scaling:
        job_name = "lammps_ani_weak_scaling"
    else:
        job_name = "lammps_ani_strong_scaling"
    setup_and_run_job(num_gpus=1, data_file=args.data_file, job_name=job_name, submit=args.y, weak_scaling=weak_scaling, log_dir=log_dir)
    setup_and_run_job(num_gpus=2, data_file=args.data_file, job_name=job_name, submit=args.y, weak_scaling=weak_scaling, log_dir=log_dir)
    setup_and_run_job(num_gpus=4, data_file=args.data_file, job_name=job_name, submit=args.y, weak_scaling=weak_scaling, log_dir=log_dir)
    setup_and_run_job(num_gpus=8, data_file=args.data_file, job_name=job_name, submit=args.y, weak_scaling=weak_scaling, log_dir=log_dir)
    setup_and_run_job(num_gpus=16, data_file=args.data_file, job_name=job_name, submit=args.y, weak_scaling=weak_scaling, log_dir=log_dir)
    setup_and_run_job(num_gpus=32, data_file=args.data_file, job_name=job_name, submit=args.y, weak_scaling=weak_scaling, log_dir=log_dir)
    setup_and_run_job(num_gpus=48, data_file=args.data_file, job_name=job_name, submit=args.y, weak_scaling=weak_scaling, log_dir=log_dir)
    setup_and_run_job(num_gpus=56, data_file=args.data_file, job_name=job_name, submit=args.y, weak_scaling=weak_scaling, log_dir=log_dir)
    setup_and_run_job(num_gpus=64, data_file=args.data_file, job_name=job_name, submit=args.y, weak_scaling=weak_scaling, log_dir=log_dir)


if __name__ == "__main__":
    main()
