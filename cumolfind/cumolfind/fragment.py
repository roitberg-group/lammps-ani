import torch
import pickle
import ase
import cudf
import cupy
import cugraph as cnx
import pandas as pd
# import mdtraj as md   # Nick NOTE: i think this is only used by molfind.py or trace.py so might not need the import
import numpy as np
import networkx as nx
import time as timetime
from torchani.neighbors import _parse_neighborlist
import matplotlib.pyplot as plt

# TODO: use RMM allocator for pytorch

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

# Check if CUDA is available
if not torch.cuda.is_available():
    raise SystemError(
        "CUDA is required to run this analysis. Please ensure a CUDA-capable GPU is available.")

# Set device to CUDA
device = "cuda"

# initial_molecules = {'HH', 'CHHHH', 'CO', 'HHHN', 'HHO', 'C', 'H', 'N', 'O'}


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
    bond_length_table = -1.0 * \
        torch.ones((PERIODIC_TABLE_LENGTH + 1), (PERIODIC_TABLE_LENGTH + 1))
    bond_length_table[bond_data_atomic_pairs[0],
                      bond_data_atomic_pairs[1]] = bond_data_length
    bond_length_table[bond_data_atomic_pairs[1],
                      bond_data_atomic_pairs[0]] = bond_data_length

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
    converts it into molecular fragments. Each fragment is accompanied by its chemical flatten_formula.

    Parameters:
    - atom_index12 (array-like): A 2D tensor. Each row signifies a pair of atom indices that
                                 represent a bonding or neighbor relationship.
    - species (array-like): The atomic species or numbers that correspond to the atoms.

    Returns:
    - df_per_frag (DataFrame): A DataFrame that groups atoms by molecular fragments. It includes
                               the flatten_formula and atom indices for each fragment. The columns are:
                               labels, collected_symbols, flatten_formula, atom_indices.
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

    # Grouping by labels and collecting symbols
    df_per_frag = df.groupby("labels").agg({"symbols": list}).reset_index()
    df_per_frag.columns = ["labels", "collected_symbols"]

    generate_real_formula = False

    if generate_real_formula:
        # using ase to get formula
        # Assuming you have ase.Atoms installed and available, you can get the chemical formula:
        df_per_frag["formula"] = cudf.from_pandas(
            df_per_frag.to_pandas()["collected_symbols"].apply(lambda x: ase.Atoms(x).get_chemical_formula())
        )  # 2s
    else:
        # Sort and concatenate symbols to create a flatten_formula (without using ASE)
        df_per_frag["collected_symbols"] = df_per_frag["collected_symbols"].list.sort_values(
            ascending=True, na_position="last"
        )
        df_per_frag["flatten_formula"] = df_per_frag.collected_symbols.str.join("")

    # Grouping by labels and collecting atom_index using cuDF
    df_per_frag["atom_indices"] = df.groupby("labels").agg(
        {"atom_index": list}).reset_index()["atom_index"]

    return cG, df_per_frag


def cugraph_slice_subgraph(cgraph, species, nodes):
    """
    Returns a subgraph of G, containing only the nodes in the list nodes with their edges.
    This implementation assume that the subgraph is a single connected component, so that every edges for each node is included.
    """

    offset_col, index_col, _ = cgraph.view_adj_list()

    edges = []
    start_indices = offset_col[nodes]
    end_indices = cupy.roll(cupy.array(offset_col), -1)[nodes]

    # we run this on CPU because now the graph is small and retrieving the data with this pattern will be slow on GPU
    for node, start_idx, end_idx in zip(nodes, start_indices.values_host, end_indices.get()):
        adj_nodes = index_col[start_idx:end_idx]
        for adj_node in adj_nodes.values_host:
            # Ensure no duplicate edges for undirected graphs
            if node < adj_node:
                edges.append((node, adj_node))

    df_edges = pd.DataFrame(edges, columns=["source", "target"])
    nxgraph = nx.from_pandas_edgelist(df_edges, "source", "target")

    atomic_numbers = species.flatten()[torch.tensor(nodes)].cpu().numpy()
    for node, atomic_number in zip(nodes, atomic_numbers):
        nxgraph.nodes[node]["atomic_number"] = atomic_number
    return nxgraph


def draw_netx_graph(nxgraph):
    import matplotlib.pyplot as plt

    labels = {
        node: f"{node}\n({nxgraph.nodes[node]['atomic_number']})" for node in nxgraph.nodes()}
    nx.draw(nxgraph, with_labels=True, labels=labels,
            node_color="lightblue", edge_color="gray")
    plt.savefig("graph.png")


def find_fragments(species, coordinates, cell=None, pbc=None, use_cell_list=True):
    """
    Find fragments for a single molecule.
    """
    assert torch.cuda.is_available(), "CUDA is required to run analysis"
    device = "cuda"

    if use_cell_list:
        neighborlist = _parse_neighborlist("cell_list", cutoff=2).to(device)
    else:
        neighborlist = _parse_neighborlist(
            "full_pairwise", cutoff=2).to(device)

    atom_index12, distances, _ = neighborlist(
        species, coordinates, cell=cell, pbc=pbc)

    bond_length_table = get_bond_data_table().to(device)
    spe12 = species.flatten()[atom_index12]
    atom_index12_bond_length = bond_length_table[spe12[0], spe12[1]]
    in_bond_length = (
        distances <= atom_index12_bond_length).nonzero().flatten()
    atom_index12 = atom_index12.index_select(1, in_bond_length)

    return neighborlist_to_fragment(atom_index12, species)


def build_netx_graph_from_ase(ase_mol, use_cell_list=True):
    """
    Build networkx object for a single ASE molecule.
    """
    species = torch.tensor(ase_mol.get_atomic_numbers(),
                           device=device).unsqueeze(0)
    positions = torch.tensor(ase_mol.get_positions(
    ), dtype=torch.float32, device=device).unsqueeze(0)

    cG, df_per_frag = find_fragments(
        species, positions, use_cell_list=use_cell_list)
    assert len(
        df_per_frag) == 1, f"This should be a single molecule, but df_per_frag has more than one fragments: {df_per_frag}"

    nxgraph = cugraph_slice_subgraph(
        cG, species, df_per_frag.iloc[0].to_pandas().atom_indices[0])

    return nxgraph


def update_graph_with_elements(graph):
    """Updates reference graph with element symbols based on atomic numbers."""
    for node, data in graph.nodes(data=True):
        data['element'] = species_dict[data['atomic_number']]
    return graph


def add_element_pairs_to_edges(graph):
    """Adds element pair attributes to the edges of the graph for easier comparison."""
    update_graph_with_elements(graph)
    for (node1, node2) in graph.edges():
        element1 = graph.nodes[node1]['element']
        element2 = graph.nodes[node2]['element']
        graph.edges[node1,
                    node2]['element_pair'] = '-'.join(sorted([element1, element2]))


def edge_match(edge1, edge2):
    """Simple check for element pairs to ensure correct bonding pattern in every molecule that is isomorphic with a reference grah."""
    return edge1['element_pair'] == edge2['element_pair']


def analyze_a_frame(
    mdtraj_frame, time_offset, dump_interval, timestep, stride, frame_num, mol_database, use_cell_list=True
):
    """
    filter_fragment_from_mdtraj_frame
    """
    start_total = timetime.time()
    start = timetime.time()
    positions = (
        torch.tensor(mdtraj_frame.xyz, device=device).float().view(
            1, -1, 3) * 10.0
    )  # convert to angstrom
    species = torch.tensor(
        [atom.element.atomic_number for atom in mdtraj_frame.topology.atoms], device=device
    ).unsqueeze(0)

    cell = torch.tensor(mdtraj_frame.unitcell_vectors[0], device=device) * 10.0
    pbc = torch.tensor([True, True, True], device=device)

    # calculate frame_offset using time_offset
    frame_offset = int(time_offset / (dump_interval * timestep * 1e-6))
    frame = frame_num * stride + frame_offset
    time = frame * timestep * dump_interval * 1e-6

    cG, df_per_frag = find_fragments(species, positions, cell, pbc, use_cell_list=use_cell_list)
    if timing:
        print("Time to find fragments: ", timetime.time() - start)

    start_filter = timetime.time()
    if timing:
        print("Time to filter fragment dataframe: ", timetime.time() - start_filter)
    df_molecule = pd.DataFrame(
        columns=[
            "frame",
            "local_frame",
            "formula",
            "flatten_formula",
            "smiles",
            "name",
            "atom_indices",
            "time"
        ]
    )

    start1 = timetime.time()
    for index, row in mol_database.iterrows():
        flatten_formula = row["flatten_formula"]
        # filter df_per_frag by flatten_formula
        df_this_formula = df_per_frag[df_per_frag["flatten_formula"] == flatten_formula]

        # skip if there is no fragment for this formula
        if len(df_this_formula) == 0:
            continue

        # we need all the atom_indices in df_this_formula
        atom_indices = df_this_formula.to_pandas().atom_indices
        # flatten the atom_indices
        atom_indices = np.concatenate(atom_indices.to_numpy())

        # unpickle the reference graph
        graph = pickle.loads(row["graph"])

        # get the nxgraph for this fragment
        nxgraph = cugraph_slice_subgraph(cG, species, atom_indices)

        start = timetime.time()
        match = 0
        # iterate all rows in the df_this_formula
        for fragment_index, fragment_row in df_this_formula.to_pandas().iterrows():
            # get the atom_indices for this fragment
            frag_atom_indices = fragment_row["atom_indices"]
            # get subgraph for this fragment
            fragment_graph = nxgraph.subgraph(frag_atom_indices)
            # add element pairs and check if this is isomorphic to the reference graph
            add_element_pairs_to_edges(fragment_graph)
            add_element_pairs_to_edges(graph)
            node_match = nx.isomorphism.categorical_node_match('element', '')
            edge_match = nx.isomorphism.categorical_edge_match('element_pair', '')
            gm = nx.isomorphism.GraphMatcher(graph, fragment_graph, node_match=node_match, edge_match=edge_match)
            if gm.is_isomorphic():
                df_molecule.loc[len(df_molecule)] = [
                    frame,
                    frame_num,  # local_frame
                    row["formula"],
                    flatten_formula,
                    row["smiles"],
                    row["name"],
                    frag_atom_indices,
                    time,
                ]
                match += 1
        if timing:
            print(f"    is_isomorphic {row['name']}, time: {timetime.time() - start}, total formula {len(df_this_formula)}, match {match}")

    if timing:
        print("iterate database: ", timetime.time() - start1)
        print("analyze_a_frame time: ", timetime.time() - start_total)

    df_formula = df_per_frag["flatten_formula"].value_counts().to_frame("counts").reset_index()

    df_formula["local_frame"] = frame_num
    df_formula["frame"] = frame
    df_formula["time"] = time

    return df_formula.to_pandas(), df_molecule
