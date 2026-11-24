import numpy as np
from .structures import read, cofactors, coords, residues_by_label, label, BACKBONE


def validate(motif_path, source_path, metadata_path):
    from .state import load, digest

    motif, source = read(motif_path), read(source_path)
    original = residues_by_label(source)
    clusters = cofactors(motif)
    if len(clusters) != 1:
        raise ValueError("Asymmetric unit must contain exactly one FES")
    cluster = clusters[0]
    iron = coords([a for a in cluster if a.element.upper() == "FE"])
    errors, donors, selections = [], [], {}
    for residue in motif.get_residues():
        ref = original.get(label(residue))
        if ref is None or ref.resname != residue.resname:
            errors.append(f"Source identity mismatch {label(residue)}")
            continue
        for atom in residue:
            if atom.get_occupancy() is None or atom.get_occupancy() <= 0:
                errors.append(f"Unresolved occupancy {label(residue)} {atom.name}")
            if atom.name in ref and atom.element != ref[atom.name].element:
                errors.append(f"Source element mismatch {label(residue)} {atom.name}")
            if atom.name not in ref or not np.allclose(
                atom.coord, ref[atom.name].coord, atol=0.001
            ):
                errors.append(f"Source atom mismatch {label(residue)} {atom.name}")
        if residue.id[0] == " ":
            missing = set(BACKBONE) - {a.name for a in residue}
            if missing:
                errors.append(f"Missing backbone {label(residue)} {sorted(missing)}")
            if {a.name for a in residue} != {a.name for a in ref if a.element != "H"}:
                errors.append(f"Incomplete source residue {label(residue)}")
            selections[label(residue)] = ",".join(a.name for a in residue)
    for residue in source.get_residues():
        if residue.id[0] != " ":
            continue
        for atom in residue:
            if atom.element.upper() in ("N", "O", "S"):
                distances = np.linalg.norm(iron - atom.coord, axis=1)
                if distances.min() <= 2.8:
                    donors.append(
                        dict(
                            residue=label(residue),
                            name=residue.resname,
                            atom=atom.name,
                            distances_A=distances.tolist(),
                        )
                    )
                    if label(residue) not in selections:
                        errors.append(
                            f"Source donor missing from motif {label(residue)} {atom.name}"
                        )
    metadata = load(metadata_path)
    declared_cluster = metadata["cluster"]
    if (
        label(cluster)
        != f"{declared_cluster['chain']}{declared_cluster['num']}{declared_cluster.get('insertion', '')}"
    ):
        errors.append("Cofactor metadata identity mismatch")
    for atom in metadata["cluster_atoms"]:
        if atom["name"] not in cluster or not np.allclose(
            cluster[atom["name"]].coord, atom["coordinates"], atol=0.001
        ):
            errors.append("Cofactor metadata coordinates mismatch")
    declared = {
        f"{r['chain']}{r['num']}{r.get('insertion', '')}" for r in metadata["coordinating_residues"]
    }
    if declared != {r["residue"] for r in donors}:
        errors.append("Legacy donor metadata disagrees with source discovery")
    if errors:
        raise ValueError("Motif validation failed: " + "; ".join(errors))
    core = coords(cluster)
    return dict(
        valid=True,
        motif_sha256=digest(motif_path),
        source_sha256=digest(source_path),
        donors=donors,
        atom_selections=selections,
        cluster=label(cluster),
        cluster_atoms={a.name: a.element for a in cluster},
        cofactor_pair_distances_A=np.linalg.norm(core[:, None] - core[None, :], axis=-1).tolist(),
        source=str(source_path),
        original_metadata=metadata,
    )
