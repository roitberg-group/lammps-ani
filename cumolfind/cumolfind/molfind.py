# Set RMM allocator to be used by PyTorch
import torch
import rmm.mr as mr
from rmm.allocators.torch import rmm_torch_allocator

# Configure PyTorch to use RAPIDS Memory Manager (RMM) for GPU memory management.
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

# Initialize a PoolMemoryResource with ManagedMemoryResource. This approach uses
# CUDA's unified memory, potentially helping with GPU OOM issues by allowing
# memory spillover to the system memory.
pool = mr.PoolMemoryResource(
    mr.ManagedMemoryResource(),
    maximum_pool_size=450 * 1024 * 1024 * 1024,  # 450GB TODO, make this configurable
)

# Set the pool as the current memory resource for RMM.
mr.set_current_device_resource(pool)

import argparse
import os
from pathlib import Path
import warnings
import torch
import pytraj as pt
import mdtraj as md
import pandas as pd
from tqdm import tqdm
from .top_loader import load_topology

# MA added
from .trackmol import track_mol_origin, analyze_all_frames_to_track
from .analyze_traj import analyze_all_frames, read_dcd_header, save_data

def main():
    parser = argparse.ArgumentParser(description="Trajectory analysis")

    parser.add_argument("traj_file", type=str, help="Trajectory file to be analyzed")
    parser.add_argument("top_file", type=str, help="H5 topology file to be analyzed")
    parser.add_argument("mol_pq", type=str, help="Molecule database file")

    # Task Selection
    parser.add_argument(
        "--task", type=str, choices=["analyze_trajectory", "track_molecules"], required=True,
        help="Specify the task to perform: 'analyze_trajectory' for full trajectory analysis, 'track_molecules' to track molecular formation."
    )
    parser.add_argument("--time_offset", type=float, help="Time offset for the trajectory", default=0.0)
    parser.add_argument(
        "--dump_interval", type=int, help="How many timesteps between frame dumps", default=50
    )
    parser.add_argument("--timestep", type=float, help="Timestep used in the simulation (fs)", default=0.25)
    parser.add_argument("--output_dir", type=str, help="Output directory", default="analyze")
    parser.add_argument(
        "--num_segments", type=int, default=1, help="Number of segments to divide the trajectory into"
    )
    parser.add_argument("--segment_index", type=int, default=0, help="Index of the segment to analyze")
    
    args = parser.parse_args()

    if args.task == "analyze_trajectory":
        print("Analyzing trajectory...")
        if Path(args.traj_file).suffix == ".xyz":
            warnings.warn("XYZ file does not have PBC information, please use DCD/NetCDF/LAMMPSTRJ file instead")

        output_directory = args.output_dir
        os.makedirs(output_directory, exist_ok=True)

        topology = load_topology(args.top_file)
        # Analyze the entire trajectory
        analyze_all_frames(
            topology,
            args.traj_file,
            args.time_offset,
            args.dump_interval,
            args.timestep,
            output_directory,
            args.mol_pq,
            args.num_segments,
            args.segment_index,
        )

    elif args.task == "track_molecules":
        output_directory = args.output_dir
        os.makedirs(output_directory, exist_ok=True)

        topology = load_topology(args.top_file)
        print("First analyzing trajectory...")
        # first we analyze the frames
        analyze_all_frames_to_track(
            topology,
            args.traj_file,
            args.time_offset,
            args.dump_interval,
            args.timestep,
            output_directory,
            args.mol_pq,
            args.num_segments,
            args.segment_index,
        )
        print("Finding origin of molecules...")
        # Track the origin of molecules
        track_mol_origin(
            topology,
            args.traj_file,
            args.time_offset,
            args.dump_interval,
            args.timestep,
            output_directory,
            args.mol_pq,
            args.num_segments,
            args.segment_index,
        )

if __name__ == "__main__":
    main()
