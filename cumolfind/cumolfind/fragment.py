import torch
import ase
import cudf
import cupy
import tempfile
import warnings
import argparse
import cugraph as cnx
from ase.io import read
from pathlib import Path
import pandas as pd
import mdtraj as md
import numpy as np
from torchani.neighbors import _parse_neighborlist
import matplotlib.pyplot as plt

# TODO: use RMM allocator for pytorch

PERIODIC_TABLE_LENGTH = 118
species_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}
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
use_cell_list = True


def get_bond_data_table():
    # make bond length longer in case it is stretched
    # TODO or use 0.15
    stretch_buffer = 0.2
    bond_data_stretched = {k: v + stretch_buffer for k, v in bond_data.items()}
    bond_data_atomic_pairs = [[], []]
    for atom12 in bond_data_stretched.keys():
        atom12 = ase.symbols.symbols2numbers(atom12)
        bond_data_atomic_pairs[0].append(atom12[0])
        bond_data_atomic_pairs[1].append(atom12[1])

    bond_data_atomic_pairs = torch.tensor(bond_data_atomic_pairs)
    bond_data_length = torch.tensor(list(bond_data_stretched.values()))

    # very simple way for pytorch to index
    bond_length_table = -1.0 * torch.ones(
        (PERIODIC_TABLE_LENGTH + 1), (PERIODIC_TABLE_LENGTH + 1)
    )
    bond_length_table[
        bond_data_atomic_pairs[0], bond_data_atomic_pairs[1]
    ] = bond_data_length
    bond_length_table[
        bond_data_atomic_pairs[1], bond_data_atomic_pairs[0]
    ] = bond_data_length

    # sanity check
    assert bond_length_table[1, 1] == bond_data_stretched["HH"]
    assert bond_length_table[1, 6] == bond_data_stretched["HC"]
    assert bond_length_table[1, 7] == bond_data_stretched["HN"]
    assert bond_length_table[1, 8] == bond_data_stretched["HO"]
    assert bond_length_table[6, 1] == bond_data_stretched["HC"]
    assert bond_length_table[7, 1] == bond_data_stretched["HN"]
    assert bond_length_table[8, 1] == bond_data_stretched["HO"]

    return bond_length_table


def neighborlist_to_fragment(atom_index12, species):
    """
    Transforms a neighbor list into molecular fragments.

    This function takes a neighbor list, represented by atom indices and atomic species, and 
    converts it into molecular fragments. Each fragment is accompanied by its chemical formula.

    Parameters:
    - atom_index12 (array-like): A 2D tensor. Each row signifies a pair of atom indices that 
                                 represent a bonding or neighbor relationship.
    - species (array-like): The atomic species or numbers that correspond to the atoms.

    Returns:
    - df_per_frag (DataFrame): A DataFrame that groups atoms by molecular fragments. It includes 
                               the formula and atom indices for each fragment. The columns are: 
                               labels, collected_symbols, formula, atom_indices.
    """
    # build cugraph from cudf edges
    # https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph.graph.from_cudf_edgelist#cugraph.Graph.from_cudf_edgelist
    df_edges = cudf.DataFrame(
        {
            "source": cupy.from_dlpack(torch.to_dlpack(atom_index12[0])),
            "destination": cupy.from_dlpack(torch.to_dlpack(atom_index12[1])),
        }
    )
    cG = cnx.Graph()
    cG.from_cudf_edgelist(df_edges, renumber=False)
    # run cugraph to find all connected_components, all the atoms that are connected
    # will be labeled by the same label
    df = cnx.connected_components(cG)
    breakpoint()

    atom_index = torch.from_dlpack(df["vertex"].to_dlpack())
    vertex_spe = species.flatten()[atom_index]
    df["atomic_numbers"] = cudf.from_dlpack(torch.to_dlpack(vertex_spe))
    # TODO: use ase to convert atomic numbers to symbols
    df["symbols"] = df["atomic_numbers"].map(species_dict)

    # rename "vertex" to "atom_index"
    df = df.rename(columns={"vertex": "atom_index"})

    # Grouping by labels and collecting symbols
    # df_per_frag = df.groupby('labels')['symbols'].apply(list).reset_index(name='collected_symbols')  # 1s
    df_per_frag = df.groupby("labels").agg({"symbols": list}).reset_index()
    df_per_frag.columns = ["labels", "collected_symbols"]

    if False:
        # using ase to get formula
        # Assuming you have ase.Atoms installed and available, you can get the chemical formula:
        df_per_frag["formula"] = cudf.from_pandas(
            df_per_frag.to_pandas()["collected_symbols"].apply(
                lambda x: ase.Atoms(x).get_chemical_formula()
            )
        )  # 2s
    else:
        # Sort and concatenate symbols to create a pseudo-formula (without using ASE)
        # df_per_frag['formula'] = df_per_frag['collected_symbols'].apply(lambda x: ''.join(sorted(x)))
        df_per_frag["collected_symbols"] = df_per_frag[
            "collected_symbols"
        ].list.sort_values(ascending=True, na_position="last")
        df_per_frag["formula"] = cudf.from_pandas(
            df_per_frag.to_pandas()["collected_symbols"].apply(
                lambda x: "".join(sorted(x))
            )
        )

    # Grouping by labels and collecting atom_index using cuDF
    df_per_frag["atom_indices"] = (
        df.groupby("labels").agg({"atom_index": list}).reset_index()["atom_index"]
    )

    breakpoint()

    return df_per_frag.to_pandas()


def get_netx_obj(species, coordinates, cell=None, pbc=None):
    assert torch.cuda.is_available(), "CUDA is required to run analysis"
    device = "cuda"
    if pbc is not None:
        print(f"pbc box is {pbc.tolist()}, {cell.tolist()}")

    if use_cell_list:
        neighborlist = _parse_neighborlist("cell_list", cutoff=2).to(device)
    else:
        neighborlist = _parse_neighborlist("full_pairwise", cutoff=2).to(device)

    atom_index12, distances, _ = neighborlist(
        species, coordinates, cell=cell, pbc=pbc
    )

    bond_length_table = get_bond_data_table().to(device)
    spe12 = species.flatten()[atom_index12]
    atom_index12_bond_length = bond_length_table[spe12[0], spe12[1]]
    in_bond_length = (distances <= atom_index12_bond_length).nonzero().flatten()
    atom_index12 = atom_index12.index_select(1, in_bond_length)

    df_per_atom, df_per_frag = neighborlist_to_fragment(atom_index12, species)
    interesting_formula = {"C2H5NO2": {"smiles": "NCC(=O)O", "resname": "GLY"}}

    df_per_frag_filtered = df_per_frag[
        df_per_frag["formula"].isin(interesting_formula.keys())
    ]

    breakpoint()
