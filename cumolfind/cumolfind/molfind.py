# Set RMM allocator to be used by PyTorch
import torch
from rmm.allocators.torch import rmm_torch_allocator
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

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
    top_file,
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

    stride = 1  # Currently hardcoded to 1, as we are splitting into segments
    save_interval = 20  # Interval for saving dataframes

    if Path(traj_file).suffix == ".dcd":
        total_frames = read_dcd_header(traj_file)
    else:
        traj_iterator = pt.iterload(traj_file, top=top_file)
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
    # first_frame = md.load_frame(traj_file, index=0, top=top_file)
    # cell = first_frame.unitcell_vectors * 10.0
    # print(f"pbc box cell is {cell.tolist()}")

    formula_dfs = []
    molecule_dfs = []

    frame_num = local_start_frame
    output_filename = f"{Path(traj_file).stem}_seg{segment_index:04d}of{num_segments:04d}"
    for mdtraj_frame in tqdm(
        md.iterload(traj_file, top=top_file, chunk=1, stride=stride, skip=local_start_frame),
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

        frame_num += 1
        if frame_num >= end_frame:
            break

    save_data(formula_dfs, output_dir, f"{output_filename}_formula.pq")
    save_data(molecule_dfs, output_dir, f"{output_filename}_molecule.pq")


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory")
    parser.add_argument("traj_file", type=str, help="Trajectory file to be analyzed")
    parser.add_argument("top_file", type=str, help="Topology file to be analyzed")
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

    # Analyze the entire trajectory
    analyze_all_frames(
        args.top_file,
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
