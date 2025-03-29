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
    find_fragments_nv,
    compute_fragment_edge_count,
    add_element_pairs_to_edges,
    cugraph_slice_subgraph_gpu
)
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

def save_xyz_file(directory, frame, flatten_formula, atom_indices, atom_symbols, x_coords, y_coords, z_coords):
    """
    Saves the molecule's XYZ coordinates into a text file.
    """
    os.makedirs(directory, exist_ok=True) 
    filename = os.path.join(directory, f"{flatten_formula}_frame_{frame}.xyz")
    with open(filename, "w") as f:
        f.write(f"{len(atom_symbols)}\n")  
        f.write(f"Frame {frame} - Molecule {flatten_formula}\n")  
        f.write("Atom Indices: " + " ".join(map(str, atom_indices)) + "\n")
        for idx, atom_idx in enumerate(atom_symbols):
            symbol = atom_symbols[idx] 
            x, y, z = x_coords[idx], y_coords[idx], z_coords[idx] 
            f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Saved XYZ file: {filename}")

def analyze_a_frame(
    mdtraj_frame, time_offset, dump_interval, timestep, stride, frame_num, mol_database, use_cell_list=True
):
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
    time = frame * timestep * dump_interval * 1e-6

    start_filter = timetime.time()
    if timing:
        print("Time to filter fragment dataframe: ", timetime.time() - start_filter)

    # we will be saving the relevant ones in this dataframe
    df_molecule = pd.DataFrame(
        columns=[
            "frame",
            "local_frame",
            "formula",
            "flatten_formula",
            "smiles",
            "name",
            "atom_indices",
            "symbols_ordered",
            "x_coords",
            "y_coords",
            "z_coords",
            "nxgraph",
            "time"
        ]
    )
    # todo for later is to make this a cudf dataframe right away in analyze_traj.py
    mol_database2 = mol_database
    mol_database2 = mol_database2.astype(str)
    mol_database2 = cudf.from_pandas(mol_database2)
    start1 = timetime.time()
    merged_df_per_frag = mol_database2.merge(df_per_frag, on="flatten_formula", how="inner")
    # check this!
    # create an nxgraph only for flatten_formulas that went through the filter
    global_atom_indices = np.concatenate(merged_df_per_frag["atom_indices"].to_pandas().to_numpy())
    # This function is the most costly!!!! (98% of the time is spent here)
    nxgraph = cugraph_slice_subgraph_gpu(cG, species, global_atom_indices)

    merged_df_per_frag["fragment_edge_count"] = merged_df_per_frag["atom_indices"].to_pandas().apply(
        lambda frag_atom_indices: compute_fragment_edge_count(frag_atom_indices, nxgraph))  # 0.009s
    # Throw away fragments that don't have the same number of edges as the reference graph
    merged_df_per_frag["num_edges"] = merged_df_per_frag["num_edges"].astype(int)
    filtered_df = merged_df_per_frag[merged_df_per_frag["fragment_edge_count"] == merged_df_per_frag["num_edges"]]  # 0.007s
    # From now on, we have to keep working in pandas because graph isomorphism check is not possible for cuDF/cuGraph
    graph_pandas = filtered_df["graph"].to_pandas()
    # Convert serialized strings to bytes
    graph_pandas = graph_pandas.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    reference_graphs = graph_pandas.apply(pickle.loads)
    filtered_df = filtered_df.to_pandas()
    filtered_df["reference_graph"] = reference_graphs
    positions = positions.squeeze(0)
    if timing:
        print("preprocessing: ", timetime.time() - start1)

    start1 = timetime.time()
    match = 0
    for local_frame, row in filtered_df.iterrows():
        frag_atom_indices = row["atom_indices"]
        # get subgraph for this fragment
        fragment_graph = nxgraph.subgraph(frag_atom_indices)
        graph = reference_graphs[local_frame]  # pull from preprocessed reference graph
        # add grapmatcher for cases when number of nodes and edges matches but doesn't correspond to the right molecule
        add_element_pairs_to_edges(fragment_graph)
        add_element_pairs_to_edges(graph)
        node_match = nx.isomorphism.categorical_node_match('element', '')
        edge_match = nx.isomorphism.categorical_edge_match('element_pair', '')
        gm = nx.isomorphism.GraphMatcher(graph, fragment_graph, node_match=node_match, edge_match=edge_match)
        if gm.is_isomorphic():
            df_molecule.loc[len(df_molecule)] = [
                frame_num,
                local_frame,
                row["formula"],
                row["flatten_formula"],
                row["smiles"],
                row["name"],
                row["atom_indices"],
                row["symbols_ordered"], 
                row["x_coords"],
                row["y_coords"],
                row["z_coords"],
                pickle.dumps(fragment_graph),
                time,
            ]
            match += 1
            print(f"    is_isomorphic {row['name']}, flatten_formula {row['flatten_formula']}, match {match}")

    if timing:
        print("iterate database: ", timetime.time() - start1)
    # the rest is kept the same
    # Ask, why do we need to return df_formula? # I can use this for mol tracking!
    df_formula = df_per_frag["flatten_formula"].value_counts().to_frame("counts").reset_index()
    df_formula = df_formula.rename(columns={"index": "flatten_formula"})  # Fix column name

    df_formula = df_formula.merge(
        df_per_frag[["flatten_formula", "atom_indices", "symbols_ordered", "x_coords", "y_coords", "z_coords"]], on="flatten_formula", how="left"
    )
    # Add metadata
    df_formula["local_frame"] = local_frame
    df_formula["frame"] = frame
    df_formula["time"] = time
    return df_formula.to_pandas(), df_molecule

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
):
    mol_database = pd.read_parquet(mol_pq)

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

    ########## MODIFY THIS PORTION FOR A DIFFERENT TRAJECTORY ##########
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

    formula_dfs = []
    molecule_dfs = []

    frame_num = local_start_frame
    print("frame_num is", frame_num)
    output_filename = f"{Path(traj_file).stem}_seg{segment_index:04d}of{num_segments:04d}"
    exit 
    for mdtraj_frame in tqdm(
        md.iterload(traj_file, top=topology, chunk=1, stride=stride, skip=local_start_frame),
        total=total_frames_in_segment,
    ):
        try:
            print("frame_num is", frame_num)
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

        frame_num += stride # MA modifed
        if frame_num >= end_frame:
            break

    save_data(formula_dfs, output_dir, f"{output_filename}_formula.pq")
    save_data(molecule_dfs, output_dir, f"{output_filename}_molecule.pq")

def load_frame_data(file_path, frame_number):
    # Use the filters argument to only load rows with the desired frame number
    frame_data = cudf.read_parquet(file_path, filters=[('frame', '==', frame_number)])
    print(f"Loaded frame {frame_number} with {len(frame_data)} rows")
    return frame_data

@torch.inference_mode()
def track_mol_origin(
    topology,
    traj_file,
    time_offset,
    dump_interval,
    timestep,
    output_dir,
    mol_pq,
    num_segments=1,
    segment_index=0,
    frame_stride=20,
    file_path=None,
    prev_file_path=None,
):

    # Great, now we will be iterating backwards from the current frame
    current_frame = extract_frame_number(file_path)
    # Read both in pd, pd is easy for saving files, then cudf for processing, dataframe is small
    mol_pq = pd.read_parquet(file_path)
    mol_pq["frame"] = current_frame
    mol_pq = mol_pq.drop(columns=['nxgraph']) # don't need nxgraph for now

    # First let's save the current frame's data for visualizing later
    formula_counter = defaultdict(int)
    for idx, row in mol_pq.iterrows():
        frame = row["frame"]
        flatten_formula = row["flatten_formula"]
        # We might have multiple glycines within the same frame to save
        # so I need to add a unique index to the filename
        formula_counter[flatten_formula] += 1
        file_index = formula_counter[flatten_formula] 
        atom_indices = row["atom_indices"]
        atom_symbols = row["symbols_ordered"]
        x_coords = row["x_coords"]
        y_coords = row["y_coords"]
        z_coords = row["z_coords"]

        unique_formula = f"{flatten_formula}_{file_index}"
        save_xyz_file(output_dir, frame, unique_formula, atom_indices, atom_symbols, x_coords, y_coords, z_coords)
    
    # now i will load with cudf because we will be merging common atoms
    # converting from pandas is not so smooth, dataframe is small so this is ok
    mol_pq = cudf.read_parquet(file_path)
    mol_pq["frame"] = current_frame
    mol_pq = mol_pq.drop(columns=['nxgraph', 'local_frame', 'smiles', 'formula', 'name', 'time'])

    if os.path.exists(prev_file_path):
        print(f"Loading previous frames from: {prev_file_path}")

        frame_num = 0 
        while True:
            try:
                # TODO currently we are loading the entire parquet file first and then filtering by frame
                # Need to think of a better way to do this
                prev_mol_pq_full = cudf.read_parquet(prev_file_path)
                prev_mol_pq = prev_mol_pq_full[prev_mol_pq_full["frame"] == frame_num]
                prev_mol_pq = prev_mol_pq.drop(columns=['counts', 'time'])
                if len(prev_mol_pq) == 0:
                    print(f"No more data found for frame {frame_num}.")
                    break

                # Explode mol_pq and prev_mol_pq to work with individual atom indices
                mol_pq_exploded = mol_pq.explode('atom_indices').rename(columns={'atom_indices': 'atom_index'})
                prev_mol_pq_exploded = prev_mol_pq.explode('atom_indices').rename(columns={'atom_indices': 'atom_index'})

                mol_pq_exploded['key'] = 0
                prev_mol_pq_exploded['key'] = 0

                # merging on atom_index to find common atoms
                merged = mol_pq_exploded.merge(
                    prev_mol_pq_exploded,
                    on='atom_index',
                    suffixes=('_current', '_previous')
                )

                # cudf needs strings, can't work with lists yet
                merged['symbols_ordered_previous'] = merged['symbols_ordered_previous'].astype(str)
                merged['x_coords_previous'] = merged['x_coords_previous'].astype(str)
                merged['y_coords_previous'] = merged['y_coords_previous'].astype(str)
                merged['z_coords_previous'] = merged['z_coords_previous'].astype(str)

                valid_matches = merged.groupby(['frame_previous', 'flatten_formula_previous']).agg({
                    'atom_index': 'unique',  
                    'symbols_ordered_previous': 'first',
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
                    frame = f"{row['frame_previous']}_{current_frame}"
                    flatten_formula = row["flatten_formula_previous"]
                    formula_counter[flatten_formula] += 1
                    file_index = formula_counter[flatten_formula]
                    atom_indices = row["atom_index"]
                    # convert the string back to a list for coordinates/symbols columns
                    atom_symbols = np.array(row["symbols_ordered_previous"].strip('[]').split(','), dtype=str)
                    x_coords = np.array(ast.literal_eval(row["x_coords_previous"]), dtype=float)
                    y_coords = np.array(ast.literal_eval(row["y_coords_previous"]), dtype=float)
                    z_coords = np.array(ast.literal_eval(row["z_coords_previous"]), dtype=float)
                    unique_formula = f"{flatten_formula}_{file_index}"
                    save_xyz_file(output_dir, frame, unique_formula, atom_indices, atom_symbols, x_coords, y_coords, z_coords)

            except Exception as e:
                print(f"Finished loading frames. Last successful frame read was {frame_num}.")
                print(f"Error: {e}")
                break 

            frame_num += 1