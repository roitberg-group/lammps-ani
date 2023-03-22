import torch
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
from torchani.aev.neighbors import _parse_neighborlist
import matplotlib.pyplot as plt

PERIODIC_TABLE_LENGTH = 118


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
    # TODO assert error if bond is not here
    bond_data = {"HH": 0.75, "HC": 1.09, "HO": 0.96, "CC": 1.54, "CO": 1.43, "OO": 1.48}
    # make bond length longer in case it is stretched
    bond_data = {k: v + 0.2 for k, v in bond_data.items()}
    bond_data_atomic_pairs = [[], []]
    for atom12 in bond_data.keys():
        atom12 = ase.symbols.symbols2numbers(atom12)
        bond_data_atomic_pairs[0].append(atom12[0])
        bond_data_atomic_pairs[1].append(atom12[1])

    bond_data_atomic_pairs = torch.tensor(bond_data_atomic_pairs)
    bond_data_label = atomicpair_to_bond_label(bond_data_atomic_pairs)
    bond_data_length = torch.tensor(list(bond_data.values()))

    # very simple way for pytorch to index
    bond_length_table = -1.0 * torch.ones(
        PERIODIC_TABLE_LENGTH * (PERIODIC_TABLE_LENGTH + 1)
    )
    bond_length_table[bond_data_label] = bond_data_length
    assert bond_length_table[119] == bond_data["HH"]
    assert bond_length_table[124] == bond_data["HC"]
    return bond_length_table


def atomicpair_to_bond_label(pairs):
    # make sure the first element is less than the second element in the pair
    pairs = pairs.sort(0).values
    bond_label = pairs[0] * PERIODIC_TABLE_LENGTH + pairs[1]
    return bond_label


def fragment(traj_file, batch_size, timestep, dump_interval):
    start = time.time()
    molecules = read(traj_file, index=":")
    print(
        f"finish reading '{traj_file}', total loading time: {time.time() - start:.2f} s"
    )

    assert torch.cuda.is_available(), "CUDA is required to run analysis"
    device = "cuda"

    coordinates = torch.cat(
        [torch.from_numpy(mol.get_positions()).unsqueeze(0) for mol in molecules], dim=0
    ).to(device)
    species = torch.cat(
        [torch.from_numpy(mol.get_atomic_numbers()).unsqueeze(0) for mol in molecules],
        dim=0,
    ).to(device)
    cell = torch.tensor(molecules[0].cell, device=device)
    pbc = torch.tensor(molecules[0].pbc, device=device)

    file_type = Path(traj_file).suffix
    lmpindex_element_dict = {1: 1, 2: 6, 3: 7, 4: 8, 5: 16, 6: 9, 7: 17}
    if file_type != ".xyz":
        # We need to convert lammps internal element index into correct species.
        # Reverse is important, otherwise 2 -> 6 (C), 6 -> 9 (F), which is not what we want.
        print(f"pbc box is {pbc.tolist()}")
        for lmpindex, element in reversed(lmpindex_element_dict.items()):
            mask = species == lmpindex
            species[mask] = element
    else:
        cell = None
        pbc = None

    total_frames = species.shape[0]
    atoms_per_molecule = species.shape[1]
    coordinates = coordinates.split(batch_size)
    species = species.split(batch_size)
    total_batches = len(species)
    print(
        f"finish reading, total_frames: {total_frames}, total_batches: {total_batches}"
    )
    pbar = pkbar.Pbar(name="processing fragments", target=total_frames)

    all_formula_counts = []
    for i in range(total_batches):
        coord = coordinates[i]
        spe = species[i]

        # generate neighborlist
        if batch_size > 1:
            neighborlist = _parse_neighborlist("full_pairwise", cutoff=2).to(device)
        else:
            neighborlist = _parse_neighborlist("cell_list", cutoff=2).to(device)
        atom_index12, distances, diff_vector, _ = neighborlist(
            spe, coord, cell=cell, pbc=pbc
        )

        # filter based on bond length
        bond_length_table = get_bond_data_table().to(device)
        spe12 = spe.flatten()[atom_index12]
        bond_label = atomicpair_to_bond_label(spe12)
        atom_index12_bond_length = bond_length_table[bond_label]
        in_bond_length = (distances <= atom_index12_bond_length).nonzero().flatten()
        atom_index12 = atom_index12.index_select(1, in_bond_length)
        distances = distances[in_bond_length]

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
        vertex_spe = spe.flatten()[atom_index]
        df["numbers"] = cudf.from_dlpack(torch.to_dlpack(vertex_spe))
        species_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}
        df["symbols"] = df["numbers"].map(species_dict)

        g = df.groupby("labels")
        frame = g["vertex"].agg("collect").list.get(0) // atoms_per_molecule
        # adjust frame by the offset
        frame += i * batch_size
        symbols_list = g[["symbols"]].agg("collect")["symbols"].list.sort_values()
        # we have to use pands to join the str list ...
        formula = symbols_list.to_pandas().apply("".join)
        formula = cudf.from_pandas(formula)
        df_frame_formula = cudf.DataFrame({"frame": frame, "formula": formula})
        # TODO cudf currently does not support GroupBy.value_counts
        df_frame_formula.to_pandas()
        # unique = df_frame_formula.groupby("frame")["formula"].unique()
        formula_counts = (
            df_frame_formula.to_pandas()
            .groupby("frame")["formula"]
            .value_counts()
            .to_frame("counts")
            .reset_index()
        )
        # format formula
        formula_counts["formula"] = formula_counts["formula"].apply(
            lambda x: ase.formula.Formula(x).format("hill")
        )
        # print(formula_counts)
        # print("done")
        all_formula_counts.append(formula_counts)
        pbar.update(i * batch_size + coord.shape[0] - 1)
        torch.cuda.synchronize()

    all_formula_counts = pd.concat(all_formula_counts)
    all_formula_counts["time"] = (
        all_formula_counts["frame"] * timestep * dump_interval * 1e-6
    )  # ns
    all_formula_counts.to_csv(f"analyze/{Path(traj_file).stem}.csv")
    plot(all_formula_counts, f"analyze/{Path(traj_file).stem}.png")

    # breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_file", type=str, help="trajectory file to be analyzed")
    parser.add_argument(
        "-t",
        "--timestep",
        type=float,
        help="timestep used in the simulation (fs)",
        default=0.5,
    )
    parser.add_argument(
        "-i",
        "--dump_interval",
        type=int,
        help="how many timesteps it dump once",
        default=100,
    )
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="batch size")
    args = parser.parse_args()

    print("start")
    if Path(args.traj_file).suffix == "xyz":
        warnings.warn(
            "xyz file does not have pbc information, please use nc file instead"
        )
    fragment(
        args.traj_file,
        batch_size=args.batch_size,
        timestep=args.timestep,
        dump_interval=args.dump_interval,
    )
