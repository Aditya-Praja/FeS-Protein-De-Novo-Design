from pathlib import Path
from copy import deepcopy
import shlex
from .structures import protein, cofactors, write, CHAIN_IDS
from .state import dump
from .rfd3 import LIGANDMPNN_REVISION


def export(structure, mapping, fixed, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    chains = [c for c in structure[0] if protein(c)]
    if len(chains) > len(CHAIN_IDS):
        raise ValueError("LigandMPNN PDB export exceeds 62 chains, reduce finite layer count")
    positions = [[r.id[1] for r in protein(c)] for c in chains]
    if not positions or any(v != positions[0] for v in positions):
        raise ValueError("Equivalent chains require identical residue-position inventories")
    fixed_labels = []
    for c in chains:
        for pos in fixed[mapping[c.id]["subunit"]]:
            if (" ", pos, " ") not in c:
                raise ValueError(f"Missing fixed position {c.id}{pos}")
            fixed_labels.append(f"{c.id}{pos}")
    groups = [[f"{c.id}{pos}" for c in chains] for pos in positions[0]]
    prepared = deepcopy(structure)
    for c in prepared[0]:
        for r in protein(c):
            if f"{c.id}{r.id[1]}" not in fixed_labels:
                for a in list(r):
                    if a.name not in ("N", "CA", "C", "O"):
                        r.detach_child(a.id)
    if len(cofactors(prepared)) != len(chains):
        raise ValueError("Expected one Fe2S2 per chain")
    path = destination / "assembly.pdb"
    write(prepared, path)
    command = [
        "python",
        "${LIGANDMPNN_REPO}/run.py",
        "--model_type",
        "ligand_mpnn",
        "--checkpoint_ligand_mpnn",
        "${LIGANDMPNN_CHECKPOINT}",
        "--pdb_path",
        str(path),
        "--out_folder",
        str(destination / "future_sequences"),
        "--fixed_residues",
        " ".join(fixed_labels),
        "--symmetry_residues",
        "|".join(",".join(g) for g in groups),
        "--symmetry_weights",
        "|".join(",".join(str(1 / len(g)) for _ in g) for g in groups),
        "--ligand_mpnn_use_side_chain_context",
        "1",
    ]
    dump(
        destination / "constraints.json",
        dict(
            fixed_residues=fixed_labels,
            equivalent_positions=groups,
            chain_mapping=mapping,
            ligandmpnn_revision=LIGANDMPNN_REVISION,
            atom_context="FES HETATM FE and S elements, positive occupancy",
            command_argv=command,
        ),
    )
    template = " ".join('"' + x + '"' if x.startswith("${") else shlex.quote(x) for x in command)
    (destination / "command.txt").write_text(template + "\n")
    return str(path)
