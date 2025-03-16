import argparse
import os
from pathlib import Path
import warnings
import torch
import pytraj as pt
import mdtraj as md
import cudf
import cupy as cp
import pandas as pd
from .analyze_traj import read_dcd_header, save_data
from .fragment import find_fragments_nv, compute_fragment_edge_count, add_element_pairs_to_edges, cugraph_slice_subgraph_gpu
from tqdm import tqdm
import networkx as nx
import pickle
import time as timetime
import numpy as np
import traceback
import re

import torch
import pickle
import ase
import cudf
import cupy as cp
import ast
import cugraph as cnx
import pandas as pd
import mdtraj as md
import numpy as np
import networkx as nx
import time as timetime
import matplotlib.pyplot as plt
from ase.build import nanotube
import json

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
    match = re.search(r'frame_(\d+)_', file_path)
    return int(match.group(1)) if match else None

def save_xyz_file(directory, frame, flatten_formula, atom_indices, atom_symbols, x_coords, y_coords, z_coords):
    """
    Saves the molecule's XYZ coordinates into a text file.
    """
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists

    filename = os.path.join(directory, f"{flatten_formula}_frame_{frame}.xyz")

    with open(filename, "w") as f:
        f.write(f"{len(atom_indices)}\n")  # Number of atoms
        f.write(f"Frame {frame} - Molecule {flatten_formula}\n")  # Comment line

        # Iterate over atom indices and their respective coordinates
        for idx, atom_idx in enumerate(atom_indices):
            symbol = atom_symbols[idx]  # Get symbol for current atom
            x, y, z = x_coords[idx], y_coords[idx], z_coords[idx]  # Extract coordinates
            f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Saved XYZ file: {filename}")

def assign_atom_symbols(atom_indices, flatten_formula):
    """
    Assigns atomic symbols based on the molecular formula.
    Ensures that the number of atoms matches the number of symbols.
    """
    atom_symbols = {}
    symbol_list = list(flatten_formula) 
    print("symbol_list: ", symbol_list)
    print("len(atom_indices): ", len(atom_indices)) 
    if len(symbol_list) != len(atom_indices):
        raise ValueError(f"Mismatch: More atoms ({len(atom_indices)}) than symbols ({len(symbol_list)}) in formula {flatten_formula}")

    for idx, atom in enumerate(atom_indices):
        atom_symbols[atom] = symbol_list[idx]  # Map atom index to its corresponding symbol
    print("atom_symbols: ", atom_symbols)
    return atom_symbols


def analyze_a_frame(
    mdtraj_frame, time_offset, dump_interval, timestep, stride, frame_num, mol_database, use_cell_list=True
):
    """
    Modfied from fragment.py for tracking molecules, save a lot more data
    """
    start = timetime.time()
    positions = (
        torch.tensor(mdtraj_frame.xyz, device="cuda").float().view(
            1, -1, 3) * 10.0 # 0.05s
    )  # convert to angstrom
    species = cp.array([atom.element.atomic_number for atom in mdtraj_frame.topology.atoms], dtype=cp.int32) # 5s
    # Convert CuPy array to a PyTorch tensor on GPU
    species = torch.as_tensor(species, device="cuda").unsqueeze(0)

    prefragment_time = timetime.time()
    print("Time to preporcess finding fragments: ", prefragment_time - start)
    fragment_time1 = timetime.time()
    cG, df_per_frag = find_fragments_nv(species, positions)
    fragment_time2 = timetime.time()
    print("Time to find fragments: ", fragment_time2 - fragment_time1)
    time_for_frame1 = timetime.time()
    # calculate frame_offset using time_offset
    frame_offset = int(time_offset / (dump_interval * timestep * 1e-6))
    frame = frame_num * stride + frame_offset
    print("frame_num", frame_num, "frame_offset", frame_offset, "frame", frame)
    time = frame * timestep * dump_interval * 1e-6
    time_for_frame2 = timetime.time()
    print("Time to calculate frame: ", time_for_frame2 - time_for_frame1)

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
    # WHY convert to pandas?
    return df_formula.to_pandas(), df_molecule
    # return df_formula, df_molecule

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

    stride = 20  # To only analyze every 20th frame
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
    print("total_frames is", total_frames)
    print("num_segments is", num_segments)
    segment_length = max(1, total_frames // num_segments)
    local_start_frame = segment_index * segment_length
    print("local_start_frame is", local_start_frame)
    # Ensure end frame does not exceed total frames
    end_frame = min(total_frames, local_start_frame + segment_length)
    print("end_frame is", end_frame)

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
):

    # Great, now we will be iterating backwards from the current frame
    # MA TODO: The following must be modified
    # For now, I am just loading the df_molecule from
    file_path = "/home/malkan/lammps-ani/cumolfind/cumolfind/test_analyze4/frame_1629_seg0000of0001_molecule.pq"
    current_frame = extract_frame_number(file_path)

    mol_pq = pd.read_parquet(file_path)
    mol_pq["frame"] = current_frame
    mol_pq["atom_indices_set"] = mol_pq["atom_indices"].apply(set)

    mol_pq["nxgraph"] = mol_pq["nxgraph"].apply(lambda x: pickle.loads(x))
    print("len(mol_pq): ", len(mol_pq))
    # print(mol_pq[['frame', 'flatten_formula', 'name', 'atom_indices']])
    # Let's only track the first molecule
    mol_pq = mol_pq.head(1)
    print("mol_pq.columns: ", mol_pq.columns)
    print("mol_pq graph", mol_pq["nxgraph"].values[0])

    previous_molecules = []
    prev_frame = current_frame - 20  

    while prev_frame > 0:  # Stop when prev_frame reaches 0 or below
        prev_file_path = f"/home/malkan/lammps-ani/cumolfind/cumolfind/test_analyze4/frame_{prev_frame}_seg0000of0001_formula.pq"

        if os.path.exists(prev_file_path):  # Check if the file exists
            print(f"Loading prev_frame: {prev_frame}")
            prev_mol_pq = pd.read_parquet(prev_file_path)
            prev_mol_pq["frame"] = prev_frame
            prev_mol_pq["atom_indices_set"] = prev_mol_pq["atom_indices"].apply(set)
            # print(prev_mol_pq[['frame', 'flatten_formula', 'atom_indices']].head(5))
            previous_molecules.append(prev_mol_pq)

        # Subtract 20 to get the next previous frame
        prev_frame -= 20  

    prev_mol_pq = pd.concat(previous_molecules, ignore_index=True)
    output_dir = "xyz_coordinates/"
    # Iterate over frames in reverse
    # MA TODO: Here check every 20 frames as opposed to one by 
    for idx, row in mol_pq.iterrows():
        frame = row["frame"]
        flatten_formula = row["flatten_formula"]
        atom_indices = row["atom_indices"]
        symbols = row["symbols_ordered"]
        x_coords = row["x_coords"]
        y_coords = row["y_coords"]
        z_coords = row["z_coords"]
        # print("atom_indices: ", atom_indices)
        # print("flatten_formula: ", flatten_formula)
        # print("symbols: ", symbols)
        # print("x_coords: ", x_coords)
        save_xyz_file(output_dir, frame, flatten_formula, atom_indices, symbols, x_coords, y_coords, z_coords)

        # Find previous frames where atoms overlap
        current_atoms = set(atom_indices) 
        current_length = len(current_atoms)
        previous_matches = prev_mol_pq[
            prev_mol_pq["atom_indices_set"].apply(lambda prev_atoms: len(current_atoms & prev_atoms) > 0)
        ]
        if not previous_matches.empty:
            for _, prev_row in previous_matches.iterrows():
                previous_frame = prev_row["frame"]
                previous_atoms = prev_row["atom_indices_set"]
                previous_formula = prev_row["flatten_formula"]
                previous_length = len(previous_atoms)
                print(f"Previous Formula: {previous_formula}, Atom Indices: {previous_atoms}")
                # Find common atoms
                common_atoms = list(current_atoms & previous_atoms)
                print("Common Atoms: ", common_atoms)

                if not common_atoms:
                    print("No common atoms found yet")
                    continue  # Skip if no common atoms
                # If the lengths of previous and current formulas are different, no graph check is needed
                if previous_length != current_length:
                    print("Skipping graph check: Different formula lengths")
                    # Save coordinates or any necessary data here
                    prev_symbols = prev_row["symbols_ordered"]
                    prev_x_coords = prev_row["x_coords"]
                    prev_y_coords = prev_row["y_coords"]
                    prev_z_coords = prev_row["z_coords"]
                    save_xyz_file(output_dir, previous_frame, previous_formula, previous_atoms, prev_symbols, prev_x_coords, prev_y_coords, prev_z_coords)
                    continue # Skip to the next iteration 
  
