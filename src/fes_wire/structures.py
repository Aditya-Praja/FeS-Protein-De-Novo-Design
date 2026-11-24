from copy import deepcopy
import gzip
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, MMCIFIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model

BACKBONE = ("N", "CA", "C", "O")
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def read(path):
    path = Path(path)
    parser = (
        MMCIFParser(QUIET=True)
        if ".cif" in path.suffixes
        else PDBParser(PERMISSIVE=False, QUIET=True)
    )
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        structure = parser.get_structure(path.name, handle)
    if len(structure) != 1:
        raise ValueError(f"Expected one model in {path}")
    atoms = list(structure.get_atoms())
    if not atoms or not np.isfinite(coords(atoms)).all():
        raise ValueError(f"Empty or nonfinite structure {path}")
    if any(a.is_disordered() or a.altloc.strip() for a in atoms):
        raise ValueError(f"Resolve alternate conformations explicitly in {path}")
    return structure


def coords(atoms):
    return np.asarray([a.coord for a in atoms], dtype=float).reshape(-1, 3)


def protein(chain):
    return [r for r in chain if r.id[0] == " "]


def cofactors(structure):
    found = []
    for r in structure.get_residues():
        if r.resname == "FES":
            expected = {"FE1": "FE", "FE2": "FE", "S1": "S", "S2": "S"}
            if {a.name: a.element.upper() for a in r} != expected:
                raise ValueError(f"Invalid FES atom inventory at {r.full_id}")
            found.append(r)
    return found


def label(residue):
    return f"{residue.parent.id}{residue.id[1]}{residue.id[2].strip()}"


def residues_by_label(structure):
    result = {}
    for r in structure.get_residues():
        key = label(r)
        if key in result:
            raise ValueError(f"Ambiguous residue label {key}")
        result[key] = r
    return result


def write(structure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pdb":
        if any(len(c.id) != 1 for c in structure.get_chains()):
            raise ValueError("PDB requires single-character chains, use mmCIF")
        io = PDBIO()
    else:
        io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(path))


def empty():
    s = Structure("assembly")
    s.add(Model(0))
    return s


def transformed(structure, rotation, translation):
    """Column-vector convention x' = R x + t, applied to every atom together."""
    result = deepcopy(structure)
    for atom in result.get_atoms():
        atom.coord = rotation @ atom.coord + translation
    return result
