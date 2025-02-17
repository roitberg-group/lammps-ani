import argparse
import os
from pathlib import Path
import warnings
import torch
import pytraj as pt
import mdtraj as md
import cudf
import pandas as pd
from tqdm import tqdm
import networkx as nx
import pickle
import re

# some helper functions
def get_frame_number(traj_file):
    filename = Path(traj_file).stem  
    match = re.search(r'frame_(\d+)', filename)  
    if match:
        return int(match.group(1))  
    else:
        raise ValueError(f"No valid frame_num found {traj_file}")

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
    file_path = "/home/malkan/test_analyze4/frame_1629_seg0000of0001_molecule.pq"
    mol_pq = pd.read_parquet(file_path)

    print("mol_pq: ", mol_pq.columns)

    mol_pq["nxgraph"] = mol_pq["nxgraph"].apply(lambda x: pickle.loads(x))

    print("len(mol_pq): ", len(mol_pq))
    # Let's only track the first molecule
    mol_pq = mol_pq.head(1)

    # New dataframe we will be saving tracked molecules to
    tracked_molecules = pd.DataFrame(
        columns=[
            "frame", 
            "atom_indices", 
            "graph"
        ]
    )

    # Convert atom indices to sets for faster lookup
    mol_pq["atom_indices_set"] = mol_pq["atom_indices"].apply(set)

    # Iterate over frames in reverse
    # MA TODO: Here check every 20 frames as opposed to one by 
    for idx, row in mol_pq.iterrows():
        current_frame = row["frame"]
        current_atoms = row["atom_indices_set"]
        current_graph = row["nxgraph"]

        # Find previous frames where atoms overlap
        previous_matches = mol_pq[
            (mol_pq["frame"] < current_frame) &  # Ensure it's a previous frame
            (mol_pq["atom_indices_set"].apply(lambda prev_atoms: len(current_atoms & prev_atoms) > 0))  # Find overlap
        ]

        if not previous_matches.empty:
            for _, prev_row in previous_matches.iterrows():
                previous_frame = prev_row["frame"]
                previous_atoms = prev_row["atom_indices_set"]
                previous_graph = prev_row["nxgraph"]

                # Find common atoms
                common_atoms = list(current_atoms & previous_atoms)

                if not common_atoms:
                    print("No common atoms found yet")
                    continue  # Skip if no common atoms

                # Ensure common atoms belong to **different graphs**
                for atom in common_atoms:
                    if previous_graph.has_node(atom) and current_graph.has_node(atom):
                        prev_neighbors = set(previous_graph.neighbors(atom))
                        curr_neighbors = set(current_graph.neighbors(atom))

                        # Check if the neighborhood has changed
                        if prev_neighbors != curr_neighbors:
                            print("Found a change in the neighborhood!")
                            # Store results
                            new_entry = cudf.DataFrame({
                                "current_frame": [current_frame],
                                "previous_frame": [previous_frame],
                                "common_atoms": [common_atoms],  
                                "new_graph": [pickle.dumps(current_graph)],  
                                "previous_graph": [pickle.dumps(previous_graph)], 
                            })

                            tracked_molecules = cudf.concat([df_tracked, new_entry], ignore_index=True)
                            break  # If we found at least one valid change, we stop checking further
                        # print("No change in the neighborhood, not saving")
    # Display the tracked molecules
    print(tracked_molecules)
  
