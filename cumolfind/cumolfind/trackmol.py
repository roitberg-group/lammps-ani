import argparse
import os
import warnings
import re
import traceback
import json
import pickle
import ast
import time as timetime
from collections import defaultdict
from pathlib import Path
import torch
import pytraj as pt
import mdtraj as md
import pandas as pd
import numpy as np
import cupy as cp
import cudf
import dask_cudf
import cugraph as cnx
import networkx as nx
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import nanotube

from .analyze_traj import read_dcd_header, save_data
from .fragment import (
    get_bond_data_table,
    compute_fragment_edge_count,
    add_element_pairs_to_edges,
    cugraph_slice_subgraph_gpu
)
# MA: For NV Cell List
from .nv_atomic_data import AtomicData
from .nv_batch import Batch
from .nv_atom_cell_list import _cell_neighbor_list

from tqdm import tqdm

timing = True
PERIODIC_TABLE_LENGTH = 118
species_dict = {1: "H", 6: "C", 7: "N", 8: "O"}
# https://media.cheggcdn.com/media%2F5fa%2F5fad12c3-ee27-47fe-917a-f7919c871c63%2FphpEjZPua.png
bond_data = {
    "HH": 0.75,
    "HC": 1.09,
    "HN": 1.01,
    "HO": 0.96,
    "CC": 1.54,
    "CN": 1.43,
    "CO": 1.43,
    "NN": 1.45,
    "NO": 1.47,
    "OO": 1.48,
}

# some helper functions
def extract_frame_number(file_path):
    match = re.search(r'frame_(\d+)\.pq', file_path)
    return int(match.group(1)) if match else None

def save_xyz_file_child(directory, frame, flatten_formula, atom_indices, atom_symbols, x_coords, y_coords, z_coords, target_id, file_index, atom_indices_buffer_str):
    """
    Saves the molecule's XYZ coordinates into a text file.
    """
    os.makedirs(directory, exist_ok=True)

    # Convert the full index buffer string to list
    full_indices = ast.literal_eval(atom_indices_buffer_str)
    if isinstance(atom_indices, (int, np.integer)):
        matched_indices = {atom_indices}
    else:
        matched_indices = set(list(atom_indices))

    filename = os.path.join(
        directory, f"{flatten_formula}_target{target_id}_{file_index}_frame_{frame}.xyz"
    )

    with open(filename, "w") as f:
        f.write(f"{len(atom_symbols)}\n")
        f.write(f"Frame {frame} - Molecule {flatten_formula} (target {target_id}, {file_index})\n")

        for idx in range(len(atom_symbols)):
            symbol = atom_symbols[idx]
            x, y, z = x_coords[idx], y_coords[idx], z_coords[idx]
            full_index = full_indices[idx]
            if full_index in matched_indices:
                f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}  # index: {full_index}\n")
            else:
                f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Saved XYZ file: {filename}")

def save_xyz_file_parent(directory, frame, flatten_formula, atom_indices, atom_symbols, x_coords, y_coords, z_coords, target_id, file_index):
    """
    Saves the molecule's XYZ coordinates into a text file.
    """
    os.makedirs(directory, exist_ok=True)

    filename = os.path.join(
        directory, f"{flatten_formula}_target{target_id}_{file_index}_frame_{frame}.xyz"
    )

    with open(filename, "w") as f:
        f.write(f"{len(atom_symbols)}\n")
        f.write(f"Frame {frame} - Molecule {flatten_formula} (target {target_id}, {file_index})\n")
        for idx in range(len(atom_symbols)):
            symbol = atom_symbols[idx]
            atom_index = atom_indices[idx]
            x, y, z = x_coords[idx], y_coords[idx], z_coords[idx]
            f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f} # index: {atom_index} \n")

    print(f"Saved XYZ file: {filename}")

def find_fragments_nv(species, coordinates):
    """
    Use NVIDIA's Cell List to find fragments for a single molecule.
    """
    device="cuda"
    cutoff = torch.tensor([2.0], device=device)

    # Compute bounding box for cell list
    eps = 1e-3
    min_ = torch.min(coordinates.view(-1, 3), dim=0)[0] - eps
    max_ = torch.max(coordinates.view(-1, 3), dim=0)[0] + eps
    largest_dist = max_ - min_
    cell = torch.eye(3, dtype=torch.float32, device=device) * largest_dist
    coordinates = coordinates - min_  
    # Set PBC explicitly to False
    pbc = torch.tensor([[False, False, False]], dtype=torch.bool, device=device)
    # Create AtomicData for full system
    atomic_data = AtomicData(
        atomic_numbers=species.squeeze(0),  
        positions=coordinates.squeeze(0),  
        cell=cell,
        pbc=pbc,
        forces=None,
        energy=None,
        charges=None,
        edge_index=None,
        node_attrs=None,
        shifts=None,
        unit_shifts=None,
        spin_multiplicity=None,
        info={},
    ) # everything until now is 0.02s
    # Compute neighbors using cell list
    # time0 = timetime.time()
    i_tensor, j_tensor, distances, coord_i, coord_j = _cell_neighbor_list(atomic_data, cutoff, max_nbins=1000000) #4.3s
    # time1 = timetime.time()
    # print("Time to compute cell list: ", time1 - time0)
    atom_index12 = torch.stack([i_tensor, j_tensor], dim=0)
    bond_length_table = get_bond_data_table().to(device)
    spe12 = species.flatten()[atom_index12]
    atom_index12_bond_length = bond_length_table[spe12[0], spe12[1]]
    in_bond_length = (
        distances <= atom_index12_bond_length).nonzero().flatten()
    atom_index12 = atom_index12.index_select(1, in_bond_length) # this bit takes 0.10s
    # for molecule tracking
    distances = distances.index_select(0, in_bond_length)  # Keep only valid distances
    coord_i = coord_i.index_select(0, in_bond_length)
    coord_j = coord_j.index_select(0, in_bond_length)
    positions = coordinates.squeeze(0)
 
    # FINDING FRAGMENTS
    # build cugraph from cudf edges
    # https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph.graph.from_cudf_edgelist#cugraph.Graph.from_cudf_edgelist
    df_edges = cudf.DataFrame(
        {
            "source": cp.from_dlpack(torch.to_dlpack(atom_index12[0])),
            "destination": cp.from_dlpack(torch.to_dlpack(atom_index12[1])),
        }
    ) #0.001s
    cG = cnx.Graph() #10^-5s
    cG.from_cudf_edgelist(df_edges[["source", "destination"]], renumber=False) 
    df = cnx.connected_components(cG)
    atom_index = torch.from_dlpack(df["vertex"].to_dlpack())
    vertex_spe = species.flatten()[atom_index]
    df["atomic_numbers"] = cudf.from_dlpack(torch.to_dlpack(vertex_spe))
    df["symbols"] = df["atomic_numbers"].map(species_dict)
    df = df.rename(columns={"vertex": "atom_index"})

    #Store x, y, z coordinates separately without converting to list
    atom_coords = positions[atom_index]
    df["x"] = cp.from_dlpack(torch.to_dlpack(atom_coords[:, 0]))  # X-coordinates
    df["y"] = cp.from_dlpack(torch.to_dlpack(atom_coords[:, 1]))  # Y-coordinates
    df["z"] = cp.from_dlpack(torch.to_dlpack(atom_coords[:, 2]))  # Z-coordinates

    #Group by labels and store separate xyz columns
    df_grouped = df.groupby("labels").agg({
        "atom_index": ["collect"],  # Collect atomic indices
        "x": ["collect"], 
        "y": ["collect"], 
        "z": ["collect"],  
    }).reset_index()

    # Rename columns for clarity
    df_grouped.columns = ["labels", "atom_indices", "x_coords", "y_coords", "z_coords"]

    df_per_frag = df.groupby("labels").agg({"symbols": list}).reset_index()

    df_per_frag.columns = ["labels", "symbols_ordered"]
    df_per_frag["collected_symbols"] = df_per_frag["symbols_ordered"].list.sort_values(
        ascending=True, na_position="last"
    )
    df_per_frag["flatten_formula"] = df_per_frag.collected_symbols.str.join("")

    if "atom_indices" in df_per_frag.columns:
        df_per_frag = df_per_frag.drop(columns=["atom_indices"])

    df_per_frag = df_per_frag.merge(df_grouped, on="labels", how="left")

    return cG, df_per_frag

def analyze_a_frame_and_track_back(
    mdtraj_frame, time_offset, dump_interval, timestep, stride, frame_num):
    """
    Modfied from fragment.py for tracking molecules, save a lot more data
    """
    # start = timetime.time()
    positions = (
        torch.tensor(mdtraj_frame.xyz, device="cuda").float().view(
            1, -1, 3) * 10.0 # 0.05s
    )  # convert to angstrom
    species = cp.array([atom.element.atomic_number for atom in mdtraj_frame.topology.atoms], dtype=cp.int32) # 5s
    # Convert CuPy array to a PyTorch tensor on GPU
    species = torch.as_tensor(species, device="cuda").unsqueeze(0)

    fragment_time1 = timetime.time()
    cG, df_per_frag = find_fragments_nv(species, positions)
    fragment_time2 = timetime.time()
    print("Time to find fragments: ", fragment_time2 - fragment_time1)
    # calculate frame_offset using time_offset
    frame_offset = int(time_offset / (dump_interval * timestep * 1e-6))
    frame = frame_num * stride + frame_offset
    df_per_frag["frame"] = frame

    return df_per_frag

def analyze_last_frame(
    mdtraj_frame, frame_to_track, frame_index):
    """
    Find the target molecule in the last frame and save it's coordinates to a new dataframe for molecule tracking
    """
    positions = (
        torch.tensor(mdtraj_frame.xyz, device="cuda").float().view(
            1, -1, 3) * 10.0 # 0.05s
    )  # convert to angstrom
    species = cp.array([atom.element.atomic_number for atom in mdtraj_frame.topology.atoms], dtype=cp.int32) # 5s
    # Convert CuPy array to a PyTorch tensor on GPU
    species = torch.as_tensor(species, device="cuda").unsqueeze(0)

    fragment_time1 = timetime.time()
    cG, df_per_frag = find_fragments_nv(species, positions)
    fragment_time2 = timetime.time()
    print("Time to find fragments: ", fragment_time2 - fragment_time1)

    # Convert atom indices in df_per_frag to sorted string keys
    atom_indices_pandas = df_per_frag["atom_indices"].to_arrow().to_pylist()
    match_keys = [str(sorted(indices)) for indices in atom_indices_pandas]
    df_per_frag = df_per_frag.assign(match_key=cudf.Series(match_keys))

    # Load frame_to_track if it's a file path
    if isinstance(frame_to_track, str):
        frame_to_track = pd.read_parquet(frame_to_track)

    # Generate all target match keys
    target_keys = [str(sorted(row["atom_indices"])) for _, row in frame_to_track.iterrows()]
    key_df = cudf.DataFrame({"match_key": target_keys})

    # Merge to find all matches
    matched_df = df_per_frag.merge(key_df, on="match_key", how="inner")

    if len(matched_df) == 0:
        raise ValueError("Target molecules you want to track are not found in this trajectory. Please modify target molecules.")

    # If needed, convert to pandas
    matched_df = matched_df.to_pandas()

    # Create a list of enriched dataframes
    saved_rows = []
    for _, row in matched_df.iterrows():
        saved_rows.append({
            "frame": frame_index,
            "flatten_formula": row["flatten_formula"],
            "atom_indices": row["atom_indices"],
            "symbols_ordered": row["symbols_ordered"],
            "x_coords": row["x_coords"],
            "y_coords": row["y_coords"],
            "z_coords": row["z_coords"],
        })

    # Combine all into a single DataFrame
    target_df = pd.DataFrame(saved_rows)
    return target_df

@torch.inference_mode()
def analyze_all_frames_to_track(
    topology,
    traj_file,
    time_offset,
    dump_interval,
    timestep,
    output_dir,
    mol_pq,
    num_segments=1,
    segment_index=0,
    stride=20,
    frame_to_track = None,
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

    formula_dfs = []
    molecule_dfs = []

    frame_num = local_start_frame
    output_filename = f"{Path(traj_file).stem}_seg{segment_index:04d}of{num_segments:04d}"
    # process the last frame first to get the target file!
    last_frame_index = total_frames
    last_mdtraj_frame = None
    for mdtraj_frame in tqdm(
        md.iterload(traj_file, top=topology, chunk=1, stride=stride, skip=local_start_frame),
        total=total_frames_in_segment,
    ):
        last_mdtraj_frame = mdtraj_frame
    # we will find the target molecule accordinat to the frame_to_track, have it's xyz saved
    mol_pq = analyze_last_frame(last_mdtraj_frame, frame_to_track, last_frame_index)
    # save the last frame's data for visualizing later
    mol_pq["target_id"] = range(len(mol_pq))
    # create a copy so we can map the atom indices with symbols ordered/coordinates
    formula_counter = defaultdict(int)
    for idx, row in mol_pq.iterrows():
        frame = row["frame"]
        target_id = row["target_id"]
        flatten_formula = row["flatten_formula"]
        # We might have multiple glycines or other targets within the same frame to save
        # so I need to add a unique index to the filename
        formula_counter[flatten_formula] += 1
        file_index = formula_counter[flatten_formula] 
        atom_indices = row["atom_indices"]
        atom_symbols = row["symbols_ordered"]
        x_coords = row["x_coords"]
        y_coords = row["y_coords"]
        z_coords = row["z_coords"]

        unique_formula = f"{flatten_formula}_{file_index}"
        save_xyz_file_parent(output_dir, frame, flatten_formula, atom_indices, atom_symbols, x_coords, y_coords, z_coords, target_id, file_index)
        
    mol_pq = cudf.from_pandas(mol_pq)
    # one less frame here since we just analyzed the last one
    for mdtraj_frame in tqdm(
        md.iterload(traj_file, top=topology, chunk=1, stride=stride, skip=local_start_frame),
        total=total_frames_in_segment,
    ):
        try:
            df_formula = analyze_a_frame_and_track_back(
                mdtraj_frame,
                time_offset,
                dump_interval,
                timestep,
                stride,
                frame_num,
            )
            # Trick here is to copy atom indices into a full buffer, we will keep the full buffer at all times while merging
            df_formula["atom_indices_buffer"] = df_formula["atom_indices"].copy()

            # Explode mol_pq and prev_mol_pq to work with individual atom indices
            mol_pq_exploded = mol_pq.explode('atom_indices').rename(columns={'atom_indices': 'atom_index'})
            mol_pq_exploded["target_id"] = mol_pq_exploded["target_id"].fillna(method='ffill')
            prev_mol_pq_exploded = df_formula.explode('atom_indices').rename(columns={'atom_indices': 'atom_index'})
            mol_pq_exploded['key'] = 0
            prev_mol_pq_exploded['key'] = 0

            # merging on atom_index to find common atoms
            merged = mol_pq_exploded.merge(
                prev_mol_pq_exploded,
                on='atom_index',
                suffixes=('_current', '_previous')
            )
            merged = merged.sort_values(by=['frame_previous', 'flatten_formula_previous', 'atom_index'])

            merged['atom_indices_buffer'] = merged['atom_indices_buffer'].astype(str)
            merged['symbols_ordered_previous'] = merged['symbols_ordered_previous'].astype(str)
            merged['x_coords_previous'] = merged['x_coords_previous'].astype(str)
            merged['y_coords_previous'] = merged['y_coords_previous'].astype(str)
            merged['z_coords_previous'] = merged['z_coords_previous'].astype(str)

            valid_matches = merged.groupby(['target_id', 'frame_previous', 'flatten_formula_previous']).agg({
                'atom_index': 'unique',  
                'symbols_ordered_previous': 'first',
                'atom_indices_buffer': 'first',
                'x_coords_previous': 'first',
                'y_coords_previous': 'first',
                'z_coords_previous': 'first'
            }).reset_index()

            current_formulas = mol_pq["flatten_formula"].unique().to_pandas().tolist()
            # Exclude any valid match that has a formula found in current_formulas
            for f in current_formulas:
                valid_matches = valid_matches[valid_matches['flatten_formula_previous'] != f]

            if len(valid_matches) == 0:
                print(f"No valid matches found in frame {frame_num}")
                frame_num += 1
                continue
            if len(valid_matches) > 0:
                print(f"Found {len(valid_matches)} valid matches in frame {frame_num}")

            # Convert valid_matches to pandas to be able to use iterrows
            valid_matches = valid_matches.to_pandas()

            formula_counter = defaultdict(int)
            for idx, row in valid_matches.iterrows():
                target_id = row["target_id"]
                flatten_formula = row["flatten_formula_previous"]
                formula_counter[flatten_formula] += 1
                file_index = formula_counter[flatten_formula]
                atom_indices = row["atom_index"]
                # convert the string back to a list for coordinates/symbols columns
                atom_symbols = np.array(row["symbols_ordered_previous"].strip('[]').split(','), dtype=str)
                x_coords = np.array(ast.literal_eval(row["x_coords_previous"]), dtype=float)
                y_coords = np.array(ast.literal_eval(row["y_coords_previous"]), dtype=float)
                z_coords = np.array(ast.literal_eval(row["z_coords_previous"]), dtype=float)
                save_xyz_file_child(
                    output_dir,
                    frame_num,
                    flatten_formula,
                    atom_indices,
                    atom_symbols,
                    x_coords,
                    y_coords,
                    z_coords,
                    target_id,
                    file_index,
                    row["atom_indices_buffer"]  
                )

        except Exception as e:
            print(f"Finished loading frames. Last successful frame read was {frame_num}.")
            print(f"Error: {e}")
            break 

        frame_num += stride # MA modifed
        if frame_num >= end_frame:
            break