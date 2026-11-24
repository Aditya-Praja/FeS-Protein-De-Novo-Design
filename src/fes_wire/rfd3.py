from pathlib import Path
import numpy as np
from .structures import (
    read,
    write,
    coords,
    protein,
    cofactors,
    residues_by_label,
    transformed,
    empty,
)
from .geometry import fit
from .state import load

REVISION = "a13b299a05897b951ea28b65e977eebb984380bc"
LIGANDMPNN_REVISION = "26ec57ac976ade5379920dbd43c7f97a91cf82de"


def specification(cfg, validation, input_path):
    selections = validation["atom_selections"]
    # Source labels refer to one ASU. Symmetric input includes all protein and FES copies.
    return {
        cfg["run_id"]: dict(
            input=Path(input_path).name,
            ligand="FES",
            unindex=",".join(selections),
            length=cfg["length"],
            select_fixed_atoms=dict(selections, FES="ALL"),
            allow_ligand_on_existing_chain=True,
            symmetry=dict(id=f"C{cfg['order']}", is_symmetric_motif=True),
        )
    }


def discover(folder):
    return sorted(
        p
        for p in Path(folder).rglob("*")
        if p.is_file()
        and (p.suffix in (".pdb", ".cif") or p.name.endswith(".cif.gz"))
        and "_noisy_" not in p.name
        and "_denoised_" not in p.name
    )


def normalize(path, reference, validation, cfg, destination):
    stem = path.name.removesuffix(".gz").rsplit(".", 1)[0]
    sidecar = path.with_name(stem + ".json")
    metadata = load(sidecar)
    mapping = metadata.get("diffused_index_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Missing explicit RFD3 diffused_index_map")
    if metadata.get("metrics", {}).get("n_conjoined_residues", 0):
        raise ValueError("Ambiguous conjoined motif residues")
    spec = metadata.get("specification", {})
    if spec.get("symmetry", {}).get("id", "").upper() != f"C{cfg['order']}":
        raise ValueError("Output specification symmetry disagrees with run")
    structure = read(path)
    unknown = [
        r.resname
        for r in structure.get_residues()
        if r.id[0] != " " and r.resname not in ("FES", "HOH", "WAT")
    ]
    if unknown:
        raise ValueError(f"Unrecognized molecular components require explicit handling: {unknown}")
    lookup, ref_lookup = residues_by_label(structure), residues_by_label(reference)
    mapped, source_points, target_points = {}, [], []
    source_chains = list(reference[0])
    for chain in source_chains:
        targets = []
        for original, names in validation["atom_selections"].items():
            key = chain.id + original[1:]
            target = mapping.get(key)
            if not isinstance(target, str) or target not in lookup:
                raise ValueError(f"Missing or ambiguous explicit motif mapping {key}: {target}")
            residue, ref = lookup[target], ref_lookup[key]
            if residue.id[0] != " " or residue.resname != ref.resname:
                raise ValueError(f"Motif residue identity changed at {target}")
            for name in names.split(","):
                if name not in residue:
                    raise ValueError(f"Constrained atom missing {target}/{name}")
                source_points.append(ref[name].coord)
                target_points.append(residue[name].coord)
            targets.append(target)
            mapped[key] = target
        if len({lookup[t].parent.id for t in targets}) != 1:
            raise ValueError("One ASU motif maps across multiple protein chains")
    if len(set(mapped.values())) != len(mapped):
        raise ValueError("Many-to-one motif mapping")
    protein_chains = [c for c in structure[0] if protein(c)]
    if len(protein_chains) != cfg["order"]:
        raise ValueError("Incorrect number of protein subunits")
    target_chains = [
        lookup[mapped[c.id + next(iter(validation["atom_selections"]))[1:]]].parent.id
        for c in source_chains
    ]
    if len(set(target_chains)) != cfg["order"]:
        raise ValueError("ASU motifs fail to cover distinct chains")
    rotation, translation, rmsd = fit(coords_array(target_points), coords_array(source_points))
    if rmsd > cfg["max_motif_rmsd_A"]:
        raise ValueError(f"Constrained motif RMSD {rmsd:.3f} A exceeds limit")
    structure = transformed(structure, rotation, translation)
    normalized = empty()
    output_mapping, fixed = {}, {}
    for source_chain, target_chain in zip(source_chains, target_chains):
        old_chain = structure[0][target_chain]
        from Bio.PDB.Chain import Chain

        new_chain = Chain(source_chain.id)
        for ordinal, residue in enumerate(protein(old_chain), 1):
            old_label = f"{target_chain}{residue.id[1]}{residue.id[2].strip()}"
            copied = residue.copy()
            copied.id = (" ", ordinal, " ")
            new_chain.add(copied)
            output_mapping[old_label] = f"{source_chain.id}{ordinal}"
        normalized[0].add(new_chain)
        fixed[source_chain.id] = [
            int(output_mapping[mapped[source_chain.id + k[1:]]][1:])
            for k in validation["atom_selections"]
        ]
    lengths = [len(protein(c)) for c in normalized[0]]
    if len(set(lengths)) != 1:
        raise ValueError("Unequal subunit lengths prevent equivalent-position mapping")
    if len({tuple(v) for v in fixed.values()}) != 1:
        raise ValueError("Motif positions differ between equivalent subunits")
    observed = cofactors(structure)
    reconstructed = not observed
    if observed and len(observed) != cfg["order"]:
        raise ValueError("Partial cofactor inventory, refusing ambiguous reconstruction")
    used = set()
    for chain in normalized[0]:
        ref = next(r for r in cofactors(reference) if r.parent.id == chain.id)
        if observed:
            distances = [np.linalg.norm(coords(r).mean(0) - coords(ref).mean(0)) for r in observed]
            index = int(np.argmin(distances))
            if index in used or distances[index] > cfg["max_motif_rmsd_A"]:
                raise ValueError("Cofactor assignment inconsistent with aligned motif")
            used.add(index)
            cofactor = observed[index].copy()
            error = max(np.linalg.norm(cofactor[a.name].coord - a.coord) for a in ref)
            if error > cfg["max_motif_rmsd_A"]:
                raise ValueError("Cofactor atom displacement exceeds motif tolerance")
        else:
            cofactor = ref.copy()
        cofactor.id = ("H_FES", lengths[0] + 1, " ")
        chain.add(cofactor)
    write(normalized, destination)
    return dict(
        design=stem,
        source=str(path),
        metadata=str(sidecar),
        structure=str(destination),
        motif_rmsd_A=rmsd,
        fixed=fixed,
        original_to_output=mapped,
        output_to_canonical=output_mapping,
        cofactor_reconstructed=reconstructed,
        reconstruction_uncertainty_A=rmsd if reconstructed else None,
        canonical_rotation=rotation.tolist(),
        canonical_translation=translation.tolist(),
        model_metadata=metadata,
    )


def coords_array(points):
    return np.asarray(points, dtype=float)
