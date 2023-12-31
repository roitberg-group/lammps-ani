import re
import torch
import pickle
import tempfile
import pubchempy
import warnings
import networkx as nx
import pandas as pd
from ase.io import read
from ase.data import chemical_symbols
from cumolfind.fragment import build_netx_graph_from_ase

nucleobases = [
    "Adenine",
    "Guanine",
    "Cytosine",
    "Thymine",
    "Uracil"
]

simple_sugars = [
    "Glucose",
    "Fructose",
    "Ribose",
    "Deoxyribose",
    "Galactose"
]

fatty_acids = [
    "Caprylic acid",
    "Capric acid",
    "Lauric acid",
    "Myristic acid",
    "Palmitic acid"
]

amino_acids = [
    "Alanine",
    "Arginine",
    "Asparagine",
    "Aspartic Acid",
    "Cysteine",
    "Glutamine",
    "Glutamic Acid",
    "Glycine",
    "Histidine",
    "Isoleucine",
    "Leucine",
    "Lysine",
    "Methionine",
    "Phenylalanine",
    "Proline",
    "Serine",
    "Threonine",
    "Tryptophan",
    "Tyrosine",
    "Valine",
]

def all_dipeptides():
    amino_acid_peptide_names = {
        "Alanine": "Alanyl",
        "Arginine": "Arginyl",
        "Asparagine": "Asparaginyl",
        "Aspartic Acid": "Aspartyl",
        "Cysteine": "Cysteinyl-",
        "Glutamine": "Glutaminyl",
        "Glutamic Acid": "Glutamyl-",
        "Glycine": "Glycyl",
        "Histidine": "Histidyl",
        "Isoleucine": "Isoleucyl",
        "Leucine": "Leucyl",
        "Lysine": "Lysyl",
        "Methionine": "Methionyl",
        "Phenylalanine": "Phenylalanyl",
        "Proline": "Prolyl-",
        "Serine": "Seryl",
        "Threonine": "Threonyl",
        "Tryptophan": "Tryptophyl",
        "Tyrosine": "Tyrosyl",
        "Valine": "Valyl"
    }


    # Create a list to store the dipeptides
    dipeptides = []

    # Iterate over each possible pair of amino acids
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            dipeptide = amino_acid_peptide_names[aa1] + aa2
            dipeptides.append(dipeptide)

    return dipeptides


dipeptides = all_dipeptides()


# DataFrame to store molecule data
mol_data = pd.DataFrame(columns=["graph", "formula", "smiles", "name", "flatten_formula"])


def is_CHNO_only(formula):
    # Regular expression to match non-CHNO elements
    non_CHNO = re.compile(r"[^CHNO0-9]")
    return not non_CHNO.search(formula)


def generate_flatten_formula(atomic_numbers):
    """
    Generates a flattened formula from a list of atomic numbers using ASE's chemical data.

    Parameters:
    atomic_numbers (list): List of atomic numbers.

    Returns:
    str: A string representing the flattened formula.
    """
    element_symbols = [chemical_symbols[num] for num in atomic_numbers]
    return ''.join(sorted(element_symbols))


def create_graph_from_compound(compound):
    """
    Create a NetworkX graph from a PubChemPy compound object.

    Parameters:
    compound (pubchempy.Compound): The compound object from PubChemPy.

    Returns:
    networkx.Graph: A graph representing the molecular structure.
    """
    # Create a new NetworkX graph
    G = nx.Graph()

    # Add nodes (atoms) to the graph
    for atom in compound.atoms:
        G.add_node(atom.aid, atomic_number=atom.number)

    # Add edges (bonds) to the graph
    for bond in compound.bonds:
        G.add_edge(bond.aid1, bond.aid2)

    return G


def verify_graph(compound, graph):
    ref_graph = create_graph_from_compound(compound)
    assert nx.is_isomorphic(ref_graph, graph), "Graphs do not match"
    print("graph check passed")


def process_molecule(mol_name):
    # TODO: we are only using the first one, do we need to check the other ones?
    pubchem_mols = pubchempy.get_compounds(mol_name, "name")

    if len(pubchem_mols) == 0:
        warnings.warn(f"Skipping {mol_name}, no PubChem entry found")
        return None

    pubchem_mol = pubchem_mols[0]
    formula = pubchem_mol.molecular_formula

    if not is_CHNO_only(formula):
        warnings.warn(f"Skipping {mol_name} with formula {formula}, contains elements other than CHNO")
        return None

    canonical_smiles = pubchem_mol.canonical_smiles
    print(
        f"===== Processing {mol_name} with formula {formula} and Canonical SMILES: {canonical_smiles} ====="
    )

    with tempfile.NamedTemporaryFile(suffix=".sdf") as temp:
        pubchempy.download("SDF", temp.name, pubchem_mol.cid, record_type="3d", overwrite=True)
        ase_mol = read(temp.name)

    # Create NetworkX graph
    netx_graph = build_netx_graph_from_ase(ase_mol, use_cell_list=False)
    verify_graph(pubchem_mol, netx_graph)

    # Serialize netx_graph with pickle
    pickled_netx_graph = pickle.dumps(netx_graph)

    # Add data to the DataFrame
    atomic_nums = ase_mol.get_atomic_numbers()
    mol_data.loc[len(mol_data)] = [
        pickled_netx_graph,
        ase_mol.get_chemical_formula(),
        pubchem_mol.canonical_smiles,
        mol_name,
        generate_flatten_formula(atomic_nums),
    ]


interested_mols = nucleobases + simple_sugars + fatty_acids + amino_acids + dipeptides
# Process each molecule
for mol in interested_mols:
    process_molecule(mol)

# Save the DataFrame
print(mol_data)
# mol_data.to_parquet("small_molecule_data.pq")
# mol_data.to_parquet("animal_acid.pq")
mol_data.to_parquet("nucleobases-simple_sugars-fatty_acids-amino_acids-dipeptides.pq")
