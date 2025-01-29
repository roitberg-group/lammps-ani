# Set RMM allocator to be used by PyTorch
import torch
import traceback
import pickle
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
from .fragment import analyze_a_frame
from .top_loader import load_topology


def save_data(temp_dfs, output_dir, filename):
    """Concatenates and saves the given list of DataFrames."""
    if temp_dfs:
        concatenated_df = pd.concat(temp_dfs)
        concatenated_df.to_parquet(os.path.join(output_dir, filename))


def read_dcd_header(dcd_file_path):
    with open(dcd_file_path, 'rb') as file:
        file.seek(8)  # Skip magic number and version
        n_frames = int.from_bytes(file.read(4), byteorder='little')
        return n_frames


@torch.inference_mode()
def analyze_all_frames(
    topology,
    traj_file,
    time_offset,
    dump_interval,
    timestep,
    output_dir,
    mol_pq,
    num_segments=1,
    segment_index=0,
):
    mol_database = pd.read_parquet(mol_pq)
    print("Columns in mol_database:", mol_database.columns)

    if "graph" in mol_database.columns:
        print("Graph column exists. Adding num_nodes and num_edges...")

        # Initialize lists for nodes and edges
        num_nodes = []
        num_edges = []

        # Compute node and edge counts for each graph
        for graph_pickle in mol_database["graph"]:
            graph = pickle.loads(graph_pickle)  # Unpickle the graph
            num_nodes.append(graph.number_of_nodes())
            num_edges.append(graph.number_of_edges())

        # Add the new columns
        mol_database["num_nodes"] = num_nodes
        mol_database["num_edges"] = num_edges

        # Save the updated DataFrame back to Parquet for future use
        mol_database.to_parquet(mol_pq)
        print("Updated mol_database saved with num_nodes and num_edges.")
    else:
        print("Graph column does not exist in mol_database. No changes made.")

    print(mol_database[["num_edges"]].head())
    stride = 1  # Currently hardcoded to 1, as we are splitting into segments
    save_interval = 20  # Interval for saving dataframes

    if Path(traj_file).suffix == ".dcd":
        total_frames = read_dcd_header(traj_file)
    else:
        traj_iterator = pt.iterload(traj_file, top=topology)
        total_frames = len(traj_iterator)

    if segment_index >= num_segments:
        raise ValueError("segment_index must be less than num_segments")

    # Calculate the range of frames for each segment
    # Ensure at least one frame per segment
    segment_length = max(1, total_frames // num_segments)
    local_start_frame = segment_index * segment_length
    # Ensure end frame does not exceed total frames
    end_frame = min(total_frames, local_start_frame + segment_length)

    # Adjust time offset for the segment
    segment_time_offset = time_offset + local_start_frame * timestep * dump_interval * 1e-6
    total_frames_in_segment = end_frame - local_start_frame
    print("time offset is", segment_time_offset, ", segment length is", segment_length)
    print(f"Total frames: {total_frames}, total frames in segment: {total_frames_in_segment}, frame range: {local_start_frame} - {end_frame}")

    # load the first frame to get the cell size
    # first_frame = md.load_frame(traj_file, index=0, top=topology)
    # cell = first_frame.unitcell_vectors * 10.0
    # print(f"pbc box cell is {cell.tolist()}")

    formula_dfs = []
    molecule_dfs = []

    frame_num = local_start_frame
    output_filename = f"{Path(traj_file).stem}_seg{segment_index:04d}of{num_segments:04d}"
    for mdtraj_frame in tqdm(
        md.iterload(traj_file, top=topology, chunk=1, stride=stride, skip=local_start_frame),
        total=total_frames_in_segment,
    ):
        try:
            df_formula, df_molecule = analyze_a_frame(
                mdtraj_frame,
                time_offset,
                dump_interval,
                timestep,
                stride,
                frame_num,
                mol_database,
                use_cell_list=True,
            )

            # Store the DataFrame for each frame
            formula_dfs.append(df_formula)
            molecule_dfs.append(df_molecule)

            if frame_num > 0 and frame_num % save_interval == 0:
                print(f"Checkpoint save at frame {frame_num} with output filename {output_filename}")
                save_data(formula_dfs, output_dir, f"{output_filename}_formula.pq")
                save_data(molecule_dfs, output_dir, f"{output_filename}_molecule.pq")
        except Exception as e:
            print(f"Error analyzing frame {frame_num}: {e}")
            # This will print the line number and other traceback details
            traceback.print_exc()

        frame_num += 1
        if frame_num >= end_frame:
            break

    save_data(formula_dfs, output_dir, f"{output_filename}_formula.pq")
    save_data(molecule_dfs, output_dir, f"{output_filename}_molecule.pq")


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory")
    parser.add_argument("traj_file", type=str, help="Trajectory file to be analyzed")
    parser.add_argument("top_file", type=str, help="H5 topology file to be analyzed")
    parser.add_argument("mol_pq", type=str, help="Molecule database file")
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

    print("Starting analysis...")
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


if __name__ == "__main__":
    main()
