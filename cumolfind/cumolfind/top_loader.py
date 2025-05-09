import typing as tp
import time
import json
import operator
import warnings

import tables
from mdtraj import Topology
import mdtraj.core.element as elem
from mdtraj.utils.six import string_types


def load_topology(file_name: str):
    """Get the topology out from the file. This function is a slightly reworked version of the topology method from md.load

    Returns
    -------
    topology : mdtraj.Topology
        A topology object
    """
    def get_node(file_name: str, root: str, name: str) -> tp.Any:
        filters = tables.Filters(complib="zlib", shuffle=True, complevel=1)
        f = tables.open_file(file_name, mode="r", filters=filters)
        return f.get_node(root, name=name)

    try:
        raw = get_node(file_name, root="/", name="topology")[0]
        if not isinstance(raw, string_types):
            raw = raw.decode()
        topology_dict = json.loads(raw)
    except tables.NoSuchNodeError:
        return None

    topology = Topology()

    for chain_dict in sorted(topology_dict["chains"], key=operator.itemgetter("index")):
        chain = topology.add_chain()
        for residue_dict in sorted(
            chain_dict["residues"], key=operator.itemgetter("index")
        ):
            try:
                resSeq = residue_dict["resSeq"]
            except KeyError:
                resSeq = None
                warnings.warn(
                    "No resSeq info found in HDF file, defaulting to zero-based idxs"
                )
            try:
                segment_id = residue_dict["segmentID"]
            except KeyError:
                segment_id = ""
            residue = topology.add_residue(
                residue_dict["name"], chain, resSeq=resSeq, segment_id=segment_id
            )
            for atom_dict in sorted(
                residue_dict["atoms"], key=operator.itemgetter("index")
            ):
                try:
                    element = elem.get_by_symbol(atom_dict["element"])
                except KeyError:
                    element = elem.virtual
                topology.add_atom(atom_dict["name"], element, residue)

    atoms = list(topology.atoms)
    for index1, index2 in topology_dict["bonds"]:
        topology.add_bond(atoms[index1], atoms[index2])

    return topology


# Usage:
# # In your script, run:
# from top_loader import load_topology
# topology = load_topology("/red/roitberg/nick_analysis/traj0.0ns.h5")
# # Must be an HDF5 topology -- the one referenced there was created from the initial geometry of the system.

