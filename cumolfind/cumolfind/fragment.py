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
from torchani.neighbors import _parse_neighborlist

# TODO: use RMM allocator for pytorch

# MA: For NV Cell List
from .nv_atomic_data import AtomicData
from .nv_batch import Batch
from .nv_atom_cell_list import _cell_neighbor_list

timing = False

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

# NICK EDITS:
# We don't want to save molecule information for the molecules present at frame zero -- also added individual elements, since a lot of those were found
initial_molecules = {'HH', 'CHHHH', 'CO', 'HHHN', 'HHO', 'C', 'H', 'N', 'O'}


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
    # timm0 = timetime.time()
    df_edges = cudf.DataFrame(
        {
            "source": cp.from_dlpack(torch.to_dlpack(atom_index12[0])),
            "destination": cp.from_dlpack(torch.to_dlpack(atom_index12[1])),
        }
    )
    cG = cnx.Graph()
    cG.from_cudf_edgelist(df_edges, renumber=False)
    df = cnx.connected_components(cG)
    atom_index = torch.from_dlpack(df["vertex"].to_dlpack())
    vertex_spe = species.flatten()[atom_index]
    df["atomic_numbers"] = cudf.from_dlpack(torch.to_dlpack(vertex_spe))
    # TODO: use ase to convert atomic numbers to symbols
    df["symbols"] = df["atomic_numbers"].map(species_dict)
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
    end_indices = cp.roll(cp.array(offset_col), -1)[nodes]

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
    # NOTE: There was some issue with nodes matching atomic numbers here, probably from something i changed

    for node, atomic_number in zip(nodes, atomic_numbers):
        nxgraph.nodes[node]["atomic_number"] = atomic_number
    return nxgraph


def cugraph_slice_subgraph_gpu(cgraph, species, nodes):
    """
    Returns a subgraph of G, containing only the nodes in the list nodes with their edges.
    GPU version.
    """
    #nodes_np = np.asarray(nodes, dtype=np.int64)
    nodes_cp = cp.asarray(nodes, dtype=cp.int64)

    offset_col, index_col, _ = cgraph.view_adj_list()
#    nodes_cp = cp.array(nodes)

    start_indices = offset_col[nodes_cp]
#    end_indices = cp.roll(cp.array(offset_col), -1)[nodes]
    end_indices = cp.roll(cp.asarray(offset_col), -1)[nodes_cp]


    node_repeats = end_indices - start_indices
    node_expanded = cp.concatenate([cp.full((int(r),), int(v), dtype=cp.int32) for v, r in zip(nodes_cp.get(), node_repeats.get())])

    start_indices = start_indices.to_cupy()
    start_indices_expanded = cp.concatenate([
        cp.repeat(cp.array(s, dtype=cp.int32), int(r)) for s, r in zip(start_indices.get(), node_repeats.get())
    ])
    offsets = cp.arange(len(node_expanded)) - cp.concatenate([cp.full((int(r),), int(v), dtype=cp.int32) for v, r in zip((cp.cumsum(node_repeats) - node_repeats).get(), node_repeats.get())])
    adj_nodes = index_col[start_indices_expanded + offsets]
    mask = node_expanded < adj_nodes
    edges = cp.vstack((node_expanded, adj_nodes)).T
    df_edges = cudf.DataFrame(edges, columns=["source", "target"]).to_pandas()

    # Force source and target to be type(int)
    df_edges["source"] = df_edges["source"].astype(int)
    df_edges["target"] = df_edges["target"].astype(int)

    nxgraph = nx.from_pandas_edgelist(df_edges, "source", "target")
    nxgraph.add_nodes_from(int(n) for n in nodes_cp.tolist())

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
        neighborlist = _parse_neighborlist("cell_list").to(device)
    else:
        neighborlist = _parse_neighborlist(
            "all_pairs").to(device)

    start = timetime.time()
    atom_index12, distances, _ = neighborlist(1.75,             # NT: changed from 2.0 to 1.75 to reduce neighbors in each AEV that are outside of the max bond length + 0.2 A buffer 
        species, coordinates, cell=cell, pbc=pbc)
    if timing:
        print("Time to compute TORCH cell list: ", timetime.time() - start)

    bond_length_table = get_bond_data_table().to(device)
    spe12 = species.flatten()[atom_index12]
    atom_index12_bond_length = bond_length_table[spe12[0], spe12[1]]
    in_bond_length = (
        distances <= atom_index12_bond_length).nonzero().flatten()
    atom_index12 = atom_index12.index_select(1, in_bond_length)

    return neighborlist_to_fragment(atom_index12, species)

def find_fragments_nv(species, coordinates, cell=None, pbc=None, use_cell_list=True):
    """
    Use NVIDIA's Cell List to find fragments for a single molecule.
    """
    device="cuda"
    cutoff = torch.tensor([2.0], device=device)
    time0 = timetime.time()
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
    i_tensor, j_tensor, distances, _, _ = _cell_neighbor_list(atomic_data, cutoff, max_nbins=1000000) #4.3s
    atom_index12 = torch.stack([i_tensor, j_tensor], dim=0)
    time1 = timetime.time()
    print("Time to compute cell list: ", time1 - time0)

    bond_length_table = get_bond_data_table().to(device)
    spe12 = species.flatten()[atom_index12]
    atom_index12_bond_length = bond_length_table[spe12[0], spe12[1]]
    in_bond_length = (
        distances <= atom_index12_bond_length).nonzero().flatten()
    atom_index12 = atom_index12.index_select(1, in_bond_length) # this bit takes 0.10s

    return neighborlist_to_fragment(atom_index12, species)

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
    """Simple check for element pairs to ensure correct bonding pattern in every molecule that is isomorphic with a reference graph."""
    return edge1['element_pair'] == edge2['element_pair']


# NOTE: TO DO:
#  * Load all graphs, rather than iterating mol_database every time, just have them pre-loaded
#    -- maybe this would be better to do in molfind.py, so that the df isn't reloaded every time a frame is analyzed?

def compute_fragment_edge_count(frag_atom_indices, nxgraph):
    fragment_graph = nxgraph.subgraph(frag_atom_indices)
    return fragment_graph.number_of_edges()

def analyze_a_frame(
    mdtraj_frame, time_offset, dump_interval, timestep, stride, frame_num, mol_database, use_cell_list=True
):
    """
    filter_fragment_from_mdtraj_frame
    """
    start = timetime.time()
    positions = (
        torch.tensor(mdtraj_frame.xyz, device=device).float().view(
            1, -1, 3) * 10.0
    )  # convert to angstrom
    species = cp.array([atom.element.atomic_number for atom in mdtraj_frame.topology.atoms], dtype=cp.int32)
    species = torch.as_tensor(species, device="cuda").unsqueeze(0)

    cell = torch.tensor(mdtraj_frame.unitcell_vectors[0], device=device) * 10.0
    pbc = torch.tensor([True, True, True], device=device)

    if timing:
        print("Time to read data from mdtraj: ", timetime.time() - start)

    fragment_time1 = timetime.time()
    # cG, df_per_frag = find_fragments_nv(species, positions, cell, pbc, use_cell_list=use_cell_list)
    cG, df_per_frag = find_fragments(species, positions, cell, pbc, use_cell_list=use_cell_list)
    fragment_time2 = timetime.time()

    if timing:
        print("Time to find fragments: ", fragment_time2 - fragment_time1)

    start1 = timetime.time()
    # calculate frame_offset using time_offset
    frame_offset = int(time_offset / (dump_interval * timestep * 1e-6))
    frame = frame_num * stride + frame_offset
    time = frame * timestep * dump_interval * 1e-6

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
            "time"
        ]
    )
    # todo for later is to make this a cudf dataframe right away in analyze_traj.py
    # NT: Drop the graph column before converting to cuDF (having trouble converting to string)
    mol_database2 = mol_database.drop(columns='graph').copy()
    mol_database2['flatten_formula'] = mol_database2['flatten_formula'].astype(str)
    mol_database2 = cudf.from_pandas(mol_database2)

    df_per_frag["flatten_formula"] = df_per_frag["flatten_formula"].astype(str)

    merged_df_per_frag = mol_database2.merge(df_per_frag, on="flatten_formula", how="inner")

    merged_df_per_frag = mol_database2.merge(df_per_frag, on="flatten_formula", how="inner")

    # If nothing matched this frame, bail out cleanly
    if len(merged_df_per_frag) == 0:
        # still record counts per formula for bookkeeping
        df_formula = df_per_frag["flatten_formula"].value_counts().to_frame("counts").reset_index()
        df_formula["local_frame"] = frame_num
        df_formula["frame"] = frame
        df_formula["time"] = time
        return df_formula.to_pandas(), df_molecule  # df_molecule is already empty

    # create an nxgraph only for flatten_formulas that went through the filter
    global_atom_indices = np.concatenate(merged_df_per_frag["atom_indices"].to_pandas().to_numpy())

    # This function is the most costly!!!! (98% of the time is spent here)
    nxgraph = cugraph_slice_subgraph_gpu(cG, species, global_atom_indices)
    merged_df_per_frag["fragment_edge_count"] = merged_df_per_frag["atom_indices"].to_pandas().apply(
        lambda frag_atom_indices: compute_fragment_edge_count(frag_atom_indices, nxgraph))  # 0.009s

    # Throw away fragments that don't have the same number of edges as the reference graph
    filtered_df = merged_df_per_frag[merged_df_per_frag["fragment_edge_count"] == merged_df_per_frag["num_edges"]]  # 0.007s
    
    # From now on, we have to keep working in pandas because graph isomorphism check is not possible for cuDF/cuGraph
    filtered_pd = filtered_df.to_pandas()

    filtered_pd = filtered_pd.merge(
        mol_database[["flatten_formula", "name", "graph"]],
        on=["flatten_formula", "name"],
        how="left",
        validate="m:1",
    )
    
    graph_pandas = filtered_pd["graph"]
    # Convert serialized strings to bytes
    graph_pandas = graph_pandas.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    reference_graphs = graph_pandas.apply(pickle.loads)
    filtered_pd["reference_graph"] = reference_graphs
    positions = positions.squeeze(0)
    if timing:
        print("preprocessing: ", timetime.time() - start1)

    start1 = timetime.time()
    match = 0

    for local_frame, row in filtered_pd.iterrows():
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
                time,
            ]
            match += 1
            print(f"    is_isomorphic {row['name']}, flatten_formula {row['flatten_formula']}, match {match}")

    if timing:
        print("iterate database: ", timetime.time() - start1)

    start1 = timetime.time()
    df_formula = df_per_frag["flatten_formula"].value_counts().to_frame("counts").reset_index()

    df_formula["local_frame"] = frame_num
    df_formula["frame"] = frame
    df_formula["time"] = time
    df_formula = df_formula.to_pandas()
    if timing:
        print("final pieces before returning: ", timetime.time() - start1)
    return df_formula, df_molecule

