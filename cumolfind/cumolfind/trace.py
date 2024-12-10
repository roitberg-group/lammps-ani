# Set RMM allocator to be used by PyTorch
import torch
import traceback
import rmm.mr as mr
from rmm.allocators.torch import rmm_torch_allocator

# Configure PyTorch to use RAPIDS Memory Manager (RMM) for GPU memory management.
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

# Initialize a PoolMemoryResource with ManagedMemoryResource. This approach uses
# CUDA's unified memory, potentially helping with GPU OOM issues by allowing
# memory spillover to the system memory.
pool = mr.PoolMemoryResource(
    mr.ManagedMemoryResource(),
    # 450GB TODO, make this configurable
    maximum_pool_size=450 * 1024 * 1024 * 1024,
)

# Set the pool as the current memory resource for RMM.
mr.set_current_device_resource(pool)

import argparse
import os
from pathlib import Path
import warnings
import pandas as pd
import mdtraj as md
from tqdm import tqdm
from .fragment import find_fragments, cugraph_slice_subgraph, draw_netx_graph


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
def trace_molecule(
    top_file,
    traj_file,
    mol_pq,
    atom_indices,
    time_offset,
    dump_interval,
    timestep,
    segment_index,
    num_segments,
    output_dir,
    reverse=True,
):
    """Tracks the formation of a molecule by analyzing trajectory frames."""
    atom_indices = torch.tensor(atom_indices, device="cuda")
    stride = 1

    # Load molecule database
    mol_database = pd.read_parquet(mol_pq)

    # Determine total frames in the trajectory
    if Path(traj_file).suffix == ".dcd":
        total_frames = read_dcd_header(traj_file)
    else:
        traj_iterator = md.iterload(traj_file, top=top_file)
        total_frames = len(traj_iterator)

    if segment_index >= num_segments:
        raise ValueError("segment_index must be less than num_segments")

    # Calculate the range of frames for this segment
    segment_length = max(1, total_frames // num_segments)
    local_start_frame = segment_index * segment_length
    end_frame = min(total_frames, local_start_frame + segment_length)

    # Adjust time offset for this segment
    segment_time_offset = time_offset + local_start_frame * timestep * dump_interval * 1e-6
    print(f"Segment {segment_index}: Frame range {local_start_frame} - {end_frame}")
    print(f"Segment time offset: {segment_time_offset}")

    # Set up reverse or forward frame range
    frame_range = range(end_frame - 1, local_start_frame - 1, -stride) if reverse else range(local_start_frame, end_frame, stride)

    # Output file for tracking data
    output_filename = f"{Path(traj_file).stem}_trace_{'_'.join(map(str, atom_indices.tolist()))}_seg{segment_index:04d}.parquet"

    tracked_data = []

    for frame_num in tqdm(frame_range, desc="Tracing molecule"):
        try:
            mdtraj_frame = md.load_frame(traj_file, index=frame_num, top=top_file)

            # Convert MDTraj data to PyTorch tensors
            positions = (
                torch.tensor(mdtraj_frame.xyz, device="cuda").float().view(1, -1, 3) * 10.0
            )  # convert to angstrom
            species = torch.tensor(
                [atom.element.atomic_number for atom in mdtraj_frame.topology.atoms], device="cuda"
            ).unsqueeze(0)

            cell = torch.tensor(mdtraj_frame.unitcell_vectors[0], device="cuda") * 10.0
            pbc = torch.tensor([True, True, True], device="cuda")

            # Find fragments in the frame
            cG, df_per_frag = find_fragments(species, positions, cell=cell, pbc=pbc, use_cell_list=True)

            # Identify the fragment containing the specified atom indices
            fragments = []
            for atom_index in atom_indices.tolist():
                fragment = df_per_frag[df_per_frag["atom_indices"].apply(lambda x: atom_index in x)].to_dict("records")
                if fragment:
                    fragments.append(fragment[0])

            # Stop if each atom index is in a separate fragment
            if len(fragments) == len(atom_indices):
                print(f"Frame {frame_num}: Each atom index is now in a separate fragment.")
                break
            for fragment in fragments:
                # Compare the fragment with known molecules
                match_found = False
                matched_name = "Unknown"
                for _, mol_row in mol_database.iterrows():
                    if mol_row["flatten_formula"] == fragment["flatten_formula"]:
                        match_found = True
                        matched_name = mol_row["name"]
                        break

                # Collect fragment details
                fragment_graph = cugraph_slice_subgraph(cG, species, fragment["atom_indices"])
                draw_netx_graph(fragment_graph)

                tracked_data.append({
                    "frame": frame_num,
                    "time": frame_num * timestep * dump_interval * 1e-6 + segment_time_offset,  # Include segment offset
                    "atom_indices": fragment["atom_indices"],
                    "formula": fragment["flatten_formula"],
                    "name": matched_name,
                })
        except Exception as e:
            print(f"Error analyzing frame {frame_num}: {e}")
            traceback.print_exc()
            break

    # Save tracking data
    pd.DataFrame(tracked_data).to_parquet(os.path.join(output_dir, output_filename))


def main():
    parser = argparse.ArgumentParser(description="Trace molecule formation backwards from a specified frame")
    parser.add_argument("traj_file", type=str, help="Trajectory file to be analyzed")
    parser.add_argument("top_file", type=str, help="Topology file to be analyzed")
    parser.add_argument("mol_pq", type=str, help="Molecule database file")
    parser.add_argument(
        "atom_indices",
        type=int,
        nargs="+",
        help="List of atom indices to track (e.g., 1 2 3)",
    )
    parser.add_argument("start_frame", type=int, help="Local frame (0-8000) within a specified traj file to start tracing from")
    parser.add_argument("--time_offset", type=float, help="Time offset for the trajectory", default=0.0)
    parser.add_argument("--dump_interval", type=int, help="How many timesteps between frame dumps", default=50)
    parser.add_argument("--timestep", type=float, help="Timestep used in the simulation (fs)", default=0.25)
    parser.add_argument("--num_segments", type=int, default=1, help="Number of segments to divide the trajectory into")
    parser.add_argument("--segment_index", type=int, default=0, help="Index of the segment to analyze")
    parser.add_argument("--output_dir", type=str, help="Output directory", default="analyze")

    args = parser.parse_args()

    print("Starting molecule trace...")
    if Path(args.traj_file).suffix == ".xyz":
        warnings.warn(
            "XYZ file does not have PBC information, please use DCD/NetCDF/LAMMPSTRJ file instead")

    output_directory = args.output_dir
    os.makedirs(output_directory, exist_ok=True)

    # Analyze the entire trajectory
    trace_molecule(
        args.traj_file,
        args.top_file,
        args.mol_pq,
        args.atom_indices,
        args.start_frame,
        args.time_offset,
        dump_interval=args.dump_interval,
        num_segments=args.num_segments,
        timestep=args.timestep,
        segment_index=args.segment_index,
        output_dir=args.output_dir,
        reverse=not args.forward,
    )


if __name__ == "__main__":
    main()
