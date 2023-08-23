import torch
import os
import ase
import time
import pkbar
import cudf
import cupy
import warnings
import argparse
import cugraph as cnx
from ase.io import read
from pathlib import Path
import pandas as pd
import mdtraj as md
import numpy as np
from torchani.aev.neighbors import _parse_neighborlist
import matplotlib.pyplot as plt

PERIODIC_TABLE_LENGTH = 118
species_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}
# https://media.cheggcdn.com/media%2F5fa%2F5fad12c3-ee27-47fe-917a-f7919c871c63%2FphpEjZPua.png
bond_data = {"HH": 0.75, "HC": 1.09, "HN": 1.01, "HO": 0.96, "CC": 1.54, "CN": 1.43, "CO": 1.43, "NN":1.45, "NO":1.47, "OO": 1.48}


def plot(df, save_to_file=None):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    df[df["formula"] == "O2"].plot.line(x="time", y="counts", ax=ax, label="O2")
    df[df["formula"] == "CH4"].plot.line(x="time", y="counts", ax=ax, label="CH4")
    df[df["formula"] == "CO2"].plot.line(x="time", y="counts", ax=ax, label="CO2")
    df[df["formula"] == "H2O"].plot.line(x="time", y="counts", ax=ax, label="H2O")
    df[df["formula"] == "CO"].plot.line(x="time", y="counts", ax=ax, label="CO")
    # df[df["formula"] == "H"].plot.line(x="time", y="counts", ax=ax, label="H")
    # df[df["formula"] == "O"].plot.line(x="time", y="counts", ax=ax, label="O")
    plt.legend(loc="best")
    plt.ylabel("molecule counts")
    plt.xlabel("time (ns)")
    if save_to_file is not None:
        plt.savefig(fname=save_to_file)
    else:
        plt.show()


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
    # breakpoint()
    # bond_data_label = atomicpair_to_bond_label(bond_data_atomic_pairs)
    bond_data_length = torch.tensor(list(bond_data_stretched.values()))

    # very simple way for pytorch to index
    bond_length_table = -1.0 * torch.ones((PERIODIC_TABLE_LENGTH + 1), (PERIODIC_TABLE_LENGTH + 1))
    bond_length_table[bond_data_atomic_pairs[0], bond_data_atomic_pairs[1]] = bond_data_length
    bond_length_table[bond_data_atomic_pairs[1], bond_data_atomic_pairs[0]] = bond_data_length

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
    Convert a neighbor list into molecular fragments.

    Given a neighbor list represented by atom indices and atomic species, this function
    returns molecular fragments along with their chemical formulae.

    Parameters:
    - atom_index12 (array-like): A 2D tensor where each row represents a pair of atom indices
                                 indicating a bonding/neighbor relationship.
    - species (array-like): Atomic species or numbers corresponding to the atoms.

    Returns:
    - df_per_atom (DataFrame): DataFrame containing original data with an added column for molecular formula.
                              The columns are: atom_index, labels, numbers, symbols, formula
    - df_per_frag (DataFrame): DataFrame grouping atoms by molecular fragments with formula and atom indices.
                              The columns are: labels, collected_symbols, formula, atom_indices
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

    atom_index = torch.from_dlpack(df["vertex"].to_dlpack())
    vertex_spe = species.flatten()[atom_index]
    df["atomic_numbers"] = cudf.from_dlpack(torch.to_dlpack(vertex_spe))
    # TODO: use ase to convert atomic numbers to symbols
    df["symbols"] = df["atomic_numbers"].map(species_dict)

    # rename "vertex" to "atom_index"
    df = df.rename(columns={"vertex": "atom_index"})

    # convert cudf to pandas
    df = df.to_pandas()

    # Grouping by labels and collecting symbols
    df_per_frag = df.groupby('labels')['symbols'].apply(list).reset_index(name='collected_symbols')

    # Assuming you have ase.Atoms installed and available, you can get the chemical formula:
    df_per_frag['formula'] = df_per_frag['collected_symbols'].apply(lambda x: ase.Atoms(x).get_chemical_formula())

    # Grouping by labels and collecting atom_index
    df_per_frag['atom_indices'] = df.groupby('labels')['atom_index'].apply(list).reset_index(name='collected_atom_index')['collected_atom_index']

    # Merging with the original dataframe
    df_per_atom = df.merge(df_per_frag[['labels', 'formula']], on='labels', how='left')

    return df_per_atom, df_per_frag


def analyze_all_frames(top_file, traj_file, batch_size, timestep, dump_interval):
    start = time.time()
    trajectory = md.load(traj_file, top=top_file)
    trajectory = trajectory[:100]
    print(
        f"finish reading '{traj_file}', total loading time: {time.time() - start:.2f} s"
    )

    assert torch.cuda.is_available(), "CUDA is required to run analysis"
    device = "cuda"

    coordinates = torch.tensor(trajectory.xyz, device=device).float() * 10.0 # convert to angstrom
    # There is no need to convert lammps internal element index into correct element type, because we have the topology information
    species = torch.tensor([atom.element.atomic_number for atom in trajectory.topology.atoms], device=device).expand(trajectory.n_frames, -1)
    # Here, we assume that the cell information is available. If not, it might be necessary to adapt.
    cell = torch.tensor(trajectory.unitcell_vectors[0], device=device) * 10.0 # convert to angstrom
    pbc = torch.tensor([True, True, True], device=device) # Assuming periodic boundary conditions
    print(f"pbc box is {pbc.tolist()}, {cell.tolist()}")

    total_frames = species.shape[0]
    atoms_per_molecule = species.shape[1]
    coordinates = coordinates.split(batch_size)
    species = species.split(batch_size)
    total_batches = len(species)

    print(
        f"finish reading, total_frames: {total_frames}, total_batches: {total_batches}"
    )
    pbar = pkbar.Pbar(name="processing fragments", target=total_frames)

    df_per_frame_list = []
    for i in range(total_batches):
        coord = coordinates[i]
        spe = species[i]

        # generate neighborlist
        if batch_size > 1:
            neighborlist = _parse_neighborlist("full_pairwise", cutoff=2).to(device)
        else:
            neighborlist = _parse_neighborlist("full_pairwise", cutoff=2).to(device)
            # TODO switch to cell_list when the system is large
            # neighborlist = _parse_neighborlist("cell_list", cutoff=2).to(device)
        atom_index12, distances, diff_vector, _ = neighborlist(
            spe, coord, cell=cell, pbc=pbc
        )

        # filter based on bond length
        bond_length_table = get_bond_data_table().to(device)
        spe12 = spe.flatten()[atom_index12]
        atom_index12_bond_length = bond_length_table[spe12[0], spe12[1]]
        in_bond_length = (distances <= atom_index12_bond_length).nonzero().flatten()
        atom_index12 = atom_index12.index_select(1, in_bond_length)
        distances = distances[in_bond_length]
        atom_index12_bond_length = atom_index12_bond_length[in_bond_length]
        # For debugging
        # torch.set_printoptions(edgeitems=100, linewidth=100000, threshold=5000000)
        # pair_dist_bond = torch.cat([atom_index12, distances.unsqueeze(0), atom_index12_bond_length.unsqueeze(0)], dim=0)
        # print(f"before number of pairs {atom_index12.shape[1]}", pair_dist_bond.transpose(0, 1))
        # pair_dist_bond = torch.cat([atom_index12, distances.unsqueeze(0), atom_index12_bond_length.unsqueeze(0)], dim=0)
        # print(f"after number of pairs {atom_index12.shape[1]}", pair_dist_bond.transpose(0, 1))

        df_per_atom, df_per_frag = neighborlist_to_fragment(atom_index12, spe)

        df_per_frag["frame"] = df_per_frag["atom_indices"].apply(lambda x: x[0]) // atoms_per_molecule
        # adjust frame by the offset
        df_per_frag["frame"] = df_per_frag["frame"] + i * batch_size

        # The colmns for df_per_frag are: frame, formula, counts
        df_per_frame = df_per_frag.groupby("frame")["formula"].value_counts().to_frame("counts").reset_index()

        df_per_frame_list.append(df_per_frame)
        pbar.update(i * batch_size + coord.shape[0] - 1)
        torch.cuda.synchronize()

        # export to csv every 1000 frames to prevent program crash
        if (i % 1000 == 0) or (i == total_batches - 1):
            df_per_frame_list_export = pd.concat(df_per_frame_list)
            df_per_frame_list_export["time"] = (
                df_per_frame_list_export["frame"] * timestep * dump_interval * 1e-6
            )  # ns
            df_per_frame_list_export.to_csv(f"{output_directory}/{Path(traj_file).stem}.csv")

    # plot(all_formula_counts_export, f"{output_directory}/{Path(traj_file).stem}.png")
    print(f"Analysis complete and exported to {output_directory}/{Path(traj_file).stem}.csv")


def identify_glycine(mol, target_smiles):
    from rdkit import Chem
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_canonical_smiles = Chem.MolToSmiles(target_mol, canonical=True)

    fragment_copy = Chem.Mol(mol)
    fragment_noH = Chem.RemoveHs(fragment_copy)
    fragment_canonical_smiles = Chem.MolToSmiles(fragment_noH, canonical=True)
    print(target_canonical_smiles, fragment_canonical_smiles)

    return target_canonical_smiles, fragment_canonical_smiles


def generate_topology_from_frame(top_file, traj_file, frame_number):
    # Load a single frame from trajectory
    traj = md.load_frame(traj_file, index=frame_number, top=top_file)
    assert torch.cuda.is_available(), "CUDA is required to run analysis"
    device = "cuda"

    coordinates = torch.tensor(traj.xyz, device=device).float() * 10.0
    species = torch.tensor([atom.element.atomic_number for atom in traj.topology.atoms], device=device).unsqueeze(0)
    cell = torch.tensor(traj.unitcell_vectors[0], device=device) * 10.0
    pbc = torch.tensor([True, True, True], device=device)
    print(f"pbc box is {pbc.tolist()}, {cell.tolist()}")

    neighborlist = _parse_neighborlist("full_pairwise", cutoff=2).to(device)
    # breakpoint()
    atom_index12, distances, _, _ = neighborlist(species, coordinates, cell=cell, pbc=pbc)

    bond_length_table = get_bond_data_table().to(device)
    spe12 = species.flatten()[atom_index12]
    atom_index12_bond_length = bond_length_table[spe12[0], spe12[1]]
    in_bond_length = (distances <= atom_index12_bond_length).nonzero().flatten()
    atom_index12 = atom_index12.index_select(1, in_bond_length)

    # # Convert tensor bond information to a format understandable by mdtraj
    # bonds = np.array(atom_index12.cpu().transpose(0, 1))
    # # breakpoint()
    # # traj.topology = traj.topology.from_dataframe(traj.topology.to_dataframe()[0], bonds)

    df_per_atom, df_per_frag = neighborlist_to_fragment(atom_index12, species)
    interesting_formula = {"C2H5NO2": {"smiles": "NCC(O)O", "resname": "GLY"}}

    df_per_frag_filtered = df_per_frag[df_per_frag['formula'].isin(interesting_formula.keys())]
    # breakpoint()
    # Convert topology to DataFrame
    atoms_df, bonds_df = traj.topology.to_dataframe()
    num_filtered_fragments = df_per_frag_filtered.shape[0]
    for idx in range(num_filtered_fragments):
        fragment = df_per_frag_filtered.iloc[idx]
        # TODO we could resemble the molecule and check if it has the same smiles
        # identify_glycine(resembled_rdkit_mol, interesting_formula[fragment["formula"]]["smiles"])
        global_atom_indices = fragment["atom_indices"]
        # Change residue name of these atom indices
        atoms_df.loc[global_atom_indices, 'resName'] = interesting_formula[fragment["formula"]]["resname"]

    # Reconstruct topology from DataFrame
    new_top = md.Topology.from_dataframe(atoms_df, bonds_df)
    traj.topology = new_top
    return traj


def extract_multiple_frames_with_new_top(top_file, traj_file, frame_number, frame_end=None):
    # Generate the new topology from the first frame
    new_topology = generate_topology_from_frame(top_file, traj_file, frame_number)

    # Save the first frame as a PDB file
    pdb_output_file = f"{output_directory}/{Path(traj_file).stem}.frame_{frame_number}.pdb"
    traj = md.load_frame(traj_file, index=frame_number, top=new_topology)
    traj.save_pdb(pdb_output_file)
    print(f"Frame {frame_number} saved as PDB to {pdb_output_file}")

    # If frame_end is specified, save the frames as a DCD file
    if frame_end is not None:
        assert frame_end > frame_number, "frame_end must be larger than frame_number"
        dcd_output_file = f"{output_directory}/{Path(traj_file).stem}.frames_{frame_number}_to_{frame_end}.dcd"
        # Load the entire DCD trajectory with the new topology
        traj = md.load(traj_file, top=new_topology)
        traj[frame_number:frame_end+1].save(dcd_output_file)
        print(f"Frames {frame_number} to {frame_end} saved as DCD to {dcd_output_file}")


if __name__ == "__main__":
    # adding usage examples into the parser help
    # python analyze.py start.pdb traj.dcd --frame=2000 --frame-end=2060
    parser = argparse.ArgumentParser(description="Analyze trajectory", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog="Usage examples:\n1. python analyze.py start.pdb traj.dcd --frame=2000 --frame-end=2060"
                                     "\n2. python analyze.py start.pdb traj.dcd --timestep 0.5 --dump_interval 100 --batch_size 1")
    parser.add_argument("start_top_file", type=str, help="Start topology file to be analyzed")
    parser.add_argument("traj_file", type=str, help="trajectory file to be analyzed")
    parser.add_argument("-t", "--timestep", type=float, help="timestep used in the simulation (fs)", default=0.5)
    parser.add_argument("-i", "--dump_interval", type=int, help="how many timesteps it dump once", default=100)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--frame", type=int, help="Frame number to extract with bonds", default=None)
    parser.add_argument("--frame-end", type=int, help="If defined, will extract all frames from frame to frame-end", default=None)
    parser.add_argument("--output_directory", type=str, help="Output directory", default="analyze")
    args = parser.parse_args()

    print("start")
    if Path(args.traj_file).suffix == "xyz":
        warnings.warn(
            "xyz file does not have pbc information, please use dcd/netcdf/lammpstrj file instead"
        )

    output_directory = args.output_directory
    os.makedirs(output_directory, exist_ok=True)

    if args.frame is not None:
        # Extract multiple frames with new topology
        extract_multiple_frames_with_new_top(args.start_top_file, args.traj_file, args.frame, args.frame_end)
    else:
        # Analyze the entire trajectory
        analyze_all_frames(args.start_top_file,
        args.traj_file,
        batch_size=args.batch_size,
        timestep=args.timestep,
        dump_interval=args.dump_interval)
