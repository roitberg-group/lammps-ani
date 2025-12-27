import os
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
        partition="hpg-turin",
        # reservation="roitberg-phase1",
        qos="roitberg",
        account="roitberg",
        gres=gres,
        mem_per_cpu="100gb",
        time="2:00:00",
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
        "module load cuda/12.8.1  gcc/14.2.0 openmpi/5.0.7 cmake/3.21.3",
        f'export LAMMPS_ANI_ROOT="{os.environ.get("LAMMPS_ANI_ROOT")}"',
        "export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/",
        "export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/",
        # setup conda in the subshell and activate the environment
        # check issue: https://github.com/conda/conda/issues/7980
        "source $(conda info --base)/etc/profile.d/conda.sh",
        "conda activate /blue/roitberg/apps/torch28",
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
        print("")

def parse_num_gpus(num_gpus):
    # Split the string on commas and strip whitespace
    num_gpus = num_gpus.split(',')
    num_gpus = [int(x.strip()) for x in num_gpus]

    return num_gpus


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LAMMPS-ANI benchmarks.")
    parser.add_argument("data_file", type=str, help="Path to the data file.")
    parser.add_argument('-n', "--num_gpus", type=str, default="1,2,4,8,16,32,64,128", help="Create a job for each number of gpus")
    parser.add_argument('-w', '--weak_scaling', action='store_true', help='If provided, the job will be weak scaling. If not, the job will be strong scaling.')
    parser.add_argument('-j', '--job_name', default="lammps_ani", help='The job name. Default is "lammps_ani".')
    parser.add_argument('-y', action='store_true', help='If provided, the job will be submitted. If not, the job will only be prepared but not submitted.')
    parser.add_argument('-l', '--log_dir', default="logs", help='The directory to store the log files. Default is "logs".')
    args = parser.parse_args()

    list_num_gpus =parse_num_gpus(args.num_gpus)
    print("num_gpus:", list_num_gpus)
    print("weak_scaling:", args.weak_scaling)

    for num_gpus in list_num_gpus:
        setup_and_run_job(num_gpus=num_gpus, data_file=args.data_file, job_name=args.job_name, submit=args.y, weak_scaling=args.weak_scaling, log_dir=args.log_dir)

if __name__ == "__main__":
    main()
