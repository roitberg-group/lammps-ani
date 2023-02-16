import torch
import ase
import cudf
import cupy
import cugraph as cnx
from ase.io import read
from torchani.aev.neighbors import _parse_neighborlist
# TODO support batches
# TODO export as csv
# TODO plot
# TODO support xyz_file argument
# TODO map to real formula

PERIODIC_TABLE_LENGTH = 118

def get_bond_data_table():
    # TODO assert error if bond is not here
    bond_data = {"HH": 0.75, "HC": 1.09, "HO": 0.96, "CC": 1.54,  "CO": 1.43, "OO": 1.48}
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
    bond_length_table = -1.0 * torch.ones(PERIODIC_TABLE_LENGTH * (PERIODIC_TABLE_LENGTH + 1))
    bond_length_table[bond_data_label] = bond_data_length
    assert bond_length_table[119] == bond_data["HH"]
    assert bond_length_table[124] == bond_data["HC"]
    return bond_length_table


def atomicpair_to_bond_label(pairs):
    # make sure the first element is less than the second element in the pair
    pairs = pairs.sort(0).values
    bond_label = pairs[0] * PERIODIC_TABLE_LENGTH + pairs[1]
    return bond_label

def fragment(xyz_file):
    molecules = read(xyz_file, index=":")
    assert torch.cuda.is_available(), "CUDA is required to run analysis"
    device = "cuda"

    coordinates = torch.cat(
        [torch.from_numpy(mol.get_positions()).unsqueeze(0) for mol in molecules], dim=0
    ).to(device)
    species = torch.cat(
        [torch.from_numpy(mol.get_atomic_numbers()).unsqueeze(0) for mol in molecules],
        dim=0,
    ).to(device)

    print("finish reading")
    coordinates = coordinates[::20].contiguous()
    species = species[::20].contiguous()
    neighborlist = _parse_neighborlist("full_pairwise", cutoff=2).to(device)
    atom_index12, _, diff_vector, distances = neighborlist(species, coordinates)

    # filter based on bond length
    bond_length_table = get_bond_data_table().to(device)
    species12 = species.flatten()[atom_index12]
    bond_label = atomicpair_to_bond_label(species12)
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
    species = species.flatten()[atom_index]
    df["numbers"] = cudf.from_dlpack(torch.to_dlpack(species))
    species_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}
    df["symbols"] = df["numbers"].map(species_dict)

    g = df.groupby("labels")
    frame = g["vertex"].agg("collect").list.get(0) // coordinates.shape[1]
    symbols_list = g[["symbols"]].agg("collect")["symbols"].list.sort_values()
    # we have to use pands to join the str list ...
    formula = symbols_list.to_pandas().apply("".join)
    formula = cudf.from_pandas(formula)
    df_frame_formula = cudf.DataFrame({"frame": frame, "formula": formula})
    # TODO cudf currently does not support GroupBy.value_counts
    df_frame_formula.to_pandas()
    # unique = df_frame_formula.groupby("frame")["formula"].unique()
    formula_counts = df_frame_formula.to_pandas().groupby("frame")["formula"].value_counts()
    print(formula_counts)
    # to get first frame
    print(formula_counts[0])

    breakpoint()


if __name__ == "__main__":
    xyz_file = "/blue/roitberg/apps/lammps-ani/myexamples/combustion/logs/2023-02-09-231903.xyz"
    # xyz_file = "/blue/roitberg/apps/lammps-ani/myexamples/combustion/logs/0-old-combustin/combustion-2023-02-09-1239.xyz"
    # xyz_file = "/blue/roitberg/apps/lammps-ani/myexamples/combustion/logs/methane.xyz"
    print("start")
    fragment(xyz_file)
