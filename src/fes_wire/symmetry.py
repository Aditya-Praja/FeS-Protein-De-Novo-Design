from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .structures import read, empty, coords, protein, cofactors, transformed, CHAIN_IDS
from .geometry import z_rotation, contacts


def prepare(cfg):
    context, motif = read(cfg["context"]), read(cfg["motif"])
    chains = [c for c in context[0] if protein(c)]
    if len(chains) != 1 or len(list(chains[0].get_atoms())) < 30:
        raise ValueError("RFD3 symmetry alignment requires one ASU with >=30 protein atoms")
    source = read(cfg["source"])
    for residue in chains[0]:
        ref = source[0][residue.parent.id][residue.id]
        for atom in residue:
            if atom.name not in ref or not np.allclose(
                atom.coord, ref[atom.name].coord, atol=0.001
            ):
                raise ValueError("Context differs from experimental source")
    if cofactors(context):
        raise ValueError("Context must contain protein only, cofactor comes from validated motif")
    chains[0].add(deepcopy(cofactors(motif)[0]))
    origin = coords(cofactors(motif)[0]).mean(0)
    orientation = Rotation.from_euler("xyz", cfg["orientation_deg"], degrees=True).as_matrix()
    translation = np.array([cfg["radius_A"], 0.0, 0.0]) - orientation @ origin
    placed = transformed(context, orientation, translation)
    ring = empty()
    transforms = []
    for i in range(cfg["order"]):
        rotation = z_rotation(i * 360 / cfg["order"])
        chain = deepcopy(next(placed.get_chains()))
        chain.id = CHAIN_IDS[i]
        for atom in chain.get_atoms():
            atom.coord = rotation @ atom.coord
        ring[0].add(chain)
        transforms.append(
            dict(
                chain=chain.id,
                rotation=(rotation @ orientation).tolist(),
                translation=(rotation @ translation).tolist(),
            )
        )
    # RFD3 centers the full input protein COM. Match that convention explicitly.
    center = coords([a for c in ring[0] for r in protein(c) for a in r]).mean(0)
    ring = transformed(ring, np.eye(3), -center)
    for t in transforms:
        t["translation"] = (np.array(t["translation"]) - center).tolist()
    for i, a in enumerate(ring[0]):
        for b in list(ring[0])[i + 1 :]:
            count, _ = contacts(coords(a.get_atoms()), coords(b.get_atoms()), cfg["clash_A"])
            if count:
                raise ValueError(f"Prepared motifs overlap at {a.id}/{b.id}, adjust placement")
    return ring, transforms


def stack(ring, layers, rise, twist):
    """Apply T^k with Rz(k*twist) and k*rise along the canonical common z-axis."""
    if layers < 1 or rise <= 0:
        raise ValueError("Positive layer count and rise required")
    assembly, mapping = empty(), {}
    for layer in range(layers):
        copy = transformed(ring, z_rotation(layer * twist), np.array([0, 0, layer * rise]))
        for chain in list(copy[0]):
            original = chain.id
            chain.detach_parent()
            index = len(mapping)
            chain.id = CHAIN_IDS[index] if index < len(CHAIN_IDS) else f"L{layer}_{original}"
            mapping[chain.id] = dict(layer=layer, subunit=original)
            assembly[0].add(chain)
    return assembly, mapping
