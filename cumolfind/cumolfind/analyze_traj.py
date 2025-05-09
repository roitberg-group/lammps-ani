import argparse
import os
from pathlib import Path
import warnings
import torch
import traceback
import pickle
import pytraj as pt
import mdtraj as md
import pandas as pd
from tqdm import tqdm
import time as timetime
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

    if "graph" in mol_database.columns:

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
    print(f"Total frames: {total_frames}, total frames in segment: {total_frames_in_segment}, frame range: {local_start_frame} - {end_frame}")

    formula_dfs = []
    molecule_dfs = []

    frame_num = local_start_frame
    output_filename = f"{Path(traj_file).stem}_seg{segment_index:04d}of{num_segments:04d}"
    
    total_analyze_time = 0.0
    for mdtraj_frame in tqdm(
        md.iterload(traj_file, top=topology, chunk=1, stride=stride, skip=local_start_frame),
        total=total_frames_in_segment,
    ):
        try:
            if frame_num == 0:
                # Don't time the first frame
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
            else:
                t0 = timetime.time()
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
                dt = timetime.time() - t0
                total_analyze_time += dt
                print("Analyzed frame in", dt, "seconds")

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

    print(f"Total trajectory time is: ", total_analyze_time, "seconds")
    save_data(formula_dfs, output_dir, f"{output_filename}_formula.pq")
    save_data(molecule_dfs, output_dir, f"{output_filename}_molecule.pq")
