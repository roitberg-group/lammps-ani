import os
import time
import argparse
import re
from simple_slurm import Slurm


def submit_job(traj_file, top_file, mol_pq, task, time_offset, num_segments, segment_index, output_dir, submit=False):
    job_name = f"cumolfind_{os.path.splitext(os.path.basename(traj_file))[0]}_segment_{segment_index:0{len(str(num_segments))}d}_of_{num_segments}"
    output_filename = f"/red/roitberg/nick_analysis/B200_rerun/logs/{job_name}_%j.log"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    num_gpus = 1
    nodes = 1
    ntasks_per_node = num_gpus
    gres = f"gpu:{1}"
    slurm = Slurm(
        job_name=job_name,
        ntasks=num_gpus,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        cpus_per_task=1,
        partition="hpg-b200",
        qos="mingjieliu-faimm",
        account="mingjieliu-faimm",
        gres=gres,
        mem_per_cpu="64gb",
        time="10:00:00",
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
        "module load cuda/12.9.1",
        'export LAMMPS_ANI_ROOT="/red/roitberg/lammps-ani"',
        "export LAMMPS_ROOT=${LAMMPS_ANI_ROOT}/external/lammps/",
        "export LAMMPS_PLUGIN_PATH=${LAMMPS_ANI_ROOT}/build/",
        # setup conda in the subshell and activate the environment
        # check issue: https://github.com/conda/conda/issues/7980
        "source $(conda info --base)/etc/profile.d/conda.sh",
        "conda activate /red/roitberg/conda-envs/envs/ani-b200",
        "echo using python: $(which python)",
        # run the job commands
        f"cumolfind-molfind {traj_file} {top_file} {mol_pq} --task={task} --time_offset={time_offset} --dump_interval=50 --timestep=0.25 --output_dir={output_dir} --num_segments={num_segments} --segment_index={segment_index}"
    ]
    commands = "\n".join(commands)
    if submit:
        slurm.sbatch(commands, convert=False)
        # prevent submitting too fast that results in the same timestamp
        time.sleep(0.1)
    else:
        print("Job will not be submitted. The following is the job script:")
        print(str(slurm) + commands)
        print("")


def main():
    parser = argparse.ArgumentParser(description="Parallelize cumolfind analysis.")
    parser.add_argument("--traj", type=str, required=True, help="Directory containing trajectory files or a single trajectory file.")
    parser.add_argument("--top", type=str, required=True, help="Topology file.")
    parser.add_argument("--mol_pq", type=str, required=True, help="Molecule database file")
    parser.add_argument("--task", type=str, choices=["analyze_trajectory", "track_molecules"], required=True, default="analyze_trajectory")
    parser.add_argument("--output_dir", type=str, help="Output directory", default="test_analyze")
    parser.add_argument("--num_segments", type=int, required=True, help="Number of segments for each trajectory.")
    parser.add_argument('-y', action='store_true', help='If provided, the job will be submitted. If not, the job will only be prepared but not submitted.')
    args = parser.parse_args()

    # Check if traj is a file or a directory
    if os.path.isfile(args.traj):
        # If it's a file, use it as the only element in traj_files
        traj_files = [args.traj]
    else:
        # If it's a directory, read and sort trajectory files with full path
        traj_files = sorted([os.path.join(args.traj, f) for f in os.listdir(args.traj) if f.endswith('.dcd')])


    for traj_file in traj_files:
        traj_filename = os.path.basename(traj_file)

        # Extract time_offset from the filename
        match = re.search(r'_(\d+\.\d+)ns\.dcd$', traj_filename)
        if match:
            time_offset = match.group(1)
            print(f"Submitting job for {traj_filename}, time_offset={time_offset}")
            for segment_index in range(args.num_segments):
                submit_job(
                    traj_file=traj_file,  # Use traj_file which will be the full path
                    top_file=args.top,
                    mol_pq=args.mol_pq,
                    task=args.task,
                    time_offset=time_offset,
                    num_segments=args.num_segments,
                    segment_index=segment_index,
                    output_dir=args.output_dir,
                    submit=args.y,
                )

if __name__ == "__main__":
    main()
