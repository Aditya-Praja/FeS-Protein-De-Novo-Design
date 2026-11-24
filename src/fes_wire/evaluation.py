from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.vectors import calc_dihedral
from .structures import coords, protein, cofactors, BACKBONE
from .geometry import contacts, z_rotation


def evaluate_ring(structure, record, cfg):
    reasons, monomers = [], []
    chains = [c for c in structure[0] if protein(c)]
    ca_arrays = []
    for chain in chains:
        residues = protein(chain)
        missing = [(r.id[1], a) for r in residues for a in BACKBONE if a not in r]
        if missing:
            reasons.append(f"{chain.id}: missing backbone atoms {missing[:5]}")
        ca = coords([r["CA"] for r in residues if "CA" in r])
        ca_arrays.append(ca)
        breaks = int(
            np.sum(
                (np.linalg.norm(np.diff(ca, axis=0), axis=1) < 3.0)
                | (np.linalg.norm(np.diff(ca, axis=0), axis=1) > 4.5)
            )
        )
        if breaks:
            reasons.append(f"{chain.id}: {breaks} CA chain breaks")
        atoms = [a for r in residues for a in r if a.name in BACKBONE]
        residue_indices = {id(a): i for i, r in enumerate(residues) for a in r}
        pairs = cKDTree(coords(atoms)).query_pairs(cfg["clash_A"])
        clashes = sum(
            abs(residue_indices[id(atoms[i])] - residue_indices[id(atoms[j])]) > 1 for i, j in pairs
        )
        if clashes:
            reasons.append(f"{chain.id}: {clashes} nonlocal backbone clashes")
        torsions = torsion_summary(residues)
        monomers.append(
            dict(
                chain=chain.id,
                secondary_structure_torsion_proxy=torsions,
                residues=len(residues),
                missing=missing,
                chainbreaks=breaks,
                backbone_clashes=clashes,
                radius_gyration_A=float(np.sqrt(np.mean(np.sum((ca - ca.mean(0)) ** 2, axis=1)))),
                compactness_Rg_over_N_cuberoot=float(
                    np.sqrt(np.mean(np.sum((ca - ca.mean(0)) ** 2, axis=1))) / len(ca) ** (1 / 3)
                ),
            )
        )
    if len(chains) != cfg["order"]:
        reasons.append("Incorrect oligomeric state")
    deviation = None
    if ca_arrays and len({len(a) for a in ca_arrays}) == 1:
        deviation = max(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (ca_arrays[0] @ z_rotation(i * 360 / cfg["order"]).T - ca) ** 2, axis=1
                        )
                    )
                )
            )
            for i, ca in enumerate(ca_arrays)
        )
    if deviation is None or deviation > cfg["max_symmetry_rmsd_A"]:
        reasons.append("Cyclic symmetry deviation missing or exceeds tolerance")
    interfaces = []
    for a, b in combinations(chains, 2):
        aa, bb = (
            coords([x for r in protein(a) for x in r if x.name in BACKBONE]),
            coords([x for r in protein(b) for x in r if x.name in BACKBONE]),
        )
        count, minimum = contacts(aa, bb, cfg["contact_A"])
        interfaces.append(dict(chains=[a.id, b.id], contacts=count, minimum_A=minimum))
        if minimum is not None and minimum < cfg["clash_A"]:
            reasons.append(f"Interchain backbone clash {a.id}/{b.id}")
    for i, chain in enumerate(chains):
        neighbor = chains[(i + 1) % len(chains)]
        edge = next(v for v in interfaces if set(v["chains"]) == {chain.id, neighbor.id})
        if edge["contacts"] < cfg["min_ring_contacts"]:
            reasons.append(f"Insufficient circumferential contacts {chain.id}/{neighbor.id}")
    all_ca = np.concatenate(ca_arrays)
    radial = np.linalg.norm(all_ca[:, :2], axis=1)
    clusters = cofactors(structure)
    if len(clusters) != cfg["order"]:
        reasons.append("Incorrect cofactor count")
    cofactor_rows = []
    for cluster in clusters:
        chain = cluster.parent
        centers = coords(cluster).mean(0)
        bb = coords([a for c in chains for r in protein(c) for a in r if a.name in BACKBONE])
        count, distance = contacts(coords(cluster), bb, 6.0)
        if distance is not None and distance < cfg["clash_A"]:
            reasons.append(f"Cofactor-backbone clash {chain.id}")
        donors = []
        for pos in record["fixed"][chain.id]:
            r = chain[(" ", pos, " ")]
            candidate = [a for a in r if a.name not in BACKBONE and a.element in ("S", "N", "O")]
            donor_distance = None
            if candidate:
                donor_distance = float(
                    min(
                        np.linalg.norm(a.coord - fe.coord)
                        for a in candidate
                        for fe in cluster
                        if fe.element == "FE"
                    )
                )
            donors.append(dict(position=pos, nearest_Fe_A=donor_distance))
            if donor_distance is None:
                reasons.append(f"Coordination atom unavailable {chain.id}{pos}")
            elif not 1.8 <= donor_distance <= 2.8:
                reasons.append(f"Coordination distance outside screening range {chain.id}{pos}")
        cofactor_rows.append(
            dict(
                chain=chain.id,
                centroid_A=centers.tolist(),
                radial_A=float(np.linalg.norm(centers[:2])),
                backbone_contacts_within_6A=count,
                minimum_backbone_A=distance,
                donors=donors,
            )
        )
    # SASA includes only resolved atoms. Explicitly named proxy because scaffold sidechains may be absent.
    ShrakeRupley(n_points=100).compute(structure[0], level="R")
    for row, cluster in zip(cofactor_rows, clusters):
        row["resolved_atom_cofactor_SASA_A2"] = float(cluster.sasa)
    return dict(
        design=record["design"],
        reasons=reasons,
        monomers=monomers,
        interfaces=interfaces,
        cofactors=cofactor_rows,
        motif_rmsd_A=record["motif_rmsd_A"],
        symmetry_rmsd_A=deviation,
        ring_thickness_CA_A=float(np.ptp(all_ca[:, 2])),
        ring_diameter_CA_A=float(2 * radial.max()),
        central_radial_clearance_CA_A=float(radial.min()),
        secondary_structure=record["model_metadata"].get("metrics", {}),
        interface_area_A2=None,
        interface_area_status="Unavailable without a defined full-atom surface model",
    )


def evaluate_stack(assembly, mapping, cfg):
    layers = sorted({v["layer"] for v in mapping.values()})
    by_layer = {
        k: coords([a for c in assembly[0] if mapping[c.id]["layer"] == k for a in c.get_atoms()])
        for k in layers
    }
    pairs, reasons = [], []
    for i, j in combinations(layers, 2):
        count, minimum = contacts(by_layer[i], by_layer[j], cfg["contact_A"])
        pairs.append(dict(layers=[i, j], atom_contacts=count, minimum_A=minimum))
        if minimum is not None and minimum < cfg["clash_A"]:
            reasons.append(f"Interlayer clash {i}/{j}")
        if j == i + 1 and count < cfg["min_axial_contacts"]:
            reasons.append(f"Insufficient axial contacts {i}/{j}")
    return dict(
        reasons=reasons,
        layer_pairs=pairs,
        axial_contacts=min(
            p["atom_contacts"] for p in pairs if p["layers"][1] == p["layers"][0] + 1
        ),
    )


def torsion_summary(residues):
    """Ramachandran-region fractions, a local conformational proxy rather than DSSP."""
    angles = []
    for previous, current, following in zip(residues, residues[1:], residues[2:]):
        if (
            "C" not in previous
            or "N" not in following
            or any(a not in current for a in ("N", "CA", "C"))
        ):
            continue
        if (
            np.linalg.norm(previous["C"].coord - current["N"].coord) > 1.8
            or np.linalg.norm(current["C"].coord - following["N"].coord) > 1.8
        ):
            continue
        phi = np.degrees(
            calc_dihedral(
                previous["C"].get_vector(),
                current["N"].get_vector(),
                current["CA"].get_vector(),
                current["C"].get_vector(),
            )
        )
        psi = np.degrees(
            calc_dihedral(
                current["N"].get_vector(),
                current["CA"].get_vector(),
                current["C"].get_vector(),
                following["N"].get_vector(),
            )
        )
        if np.isfinite([phi, psi]).all():
            angles.append((phi, psi))
    return dict(
        measured_residues=len(angles),
        alpha_region_fraction=sum(-100 <= p <= -30 and -80 <= q <= 0 for p, q in angles)
        / len(angles)
        if angles
        else None,
        beta_region_fraction=sum(-180 <= p <= -60 and (q >= 90 or q <= -120) for p, q in angles)
        / len(angles)
        if angles
        else None,
    )
