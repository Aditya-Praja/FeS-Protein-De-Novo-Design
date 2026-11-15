#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

BB = ("N", "CA", "C", "O")


def backbone_checks(path: Path, min_ca: float, max_ca: float, min_nonlocal: float):
    reasons = []
    structure = PDBParser(QUIET=True).get_structure("candidate", str(path))
    model = structure[0]
    chains = list(model)
    if len(chains) != 1:
        reasons.append(f"expected one protein chain, found {len(chains)}")
    if not chains:
        return reasons, {}
    chain = chains[0]
    residues = [r for r in chain if r.id[0] == " "]
    if len(residues) < 5:
        reasons.append("fewer than 5 residues")
        return reasons, {}
    nums = [r.id[1] for r in residues]
    if any(r.id[2].strip() for r in residues):
        reasons.append("PDB insertion codes present: check ProteinMPNN numbering")
    if nums != list(range(nums[0], nums[0] + len(nums))):
        reasons.append("PDB residue numbers have gaps/duplicates: ProteinMPNN parser inserts gap positions")
    missing = [(r.id[1], a) for r in residues for a in BB if a not in r]
    if missing:
        reasons.append(f"missing required backbone atoms, examples {missing[:5]}")
        return reasons, {"length": len(residues)}
    for r in residues:
        if not all(np.isfinite(r[a].coord).all() for a in BB):
            reasons.append(f"nonfinite backbone coordinates at residue {r.id[1]}")
            return reasons, {"length": len(residues)}
    ca = np.stack([r["CA"].coord for r in residues])
    neighbor_dist = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
    bad = [i for i, d in enumerate(neighbor_dist, 1) if not min_ca <= d <= max_ca]
    if bad:
        reasons.append(f"{len(bad)} adjacent CA-CA distances outside [{min_ca}, {max_ca}] A; first pair starts at index {bad[0]}")
    shortest_nonlocal = None
    if len(ca) >= 5:
        # Ignore residues separated by three or fewer sequence positions.
        i, j = np.triu_indices(len(ca), k=4)
        shortest_nonlocal = float(np.min(np.linalg.norm(ca[i] - ca[j], axis=1)))
        if shortest_nonlocal < min_nonlocal:
            reasons.append(f"nonlocal CA-CA < {min_nonlocal} A: {shortest_nonlocal:.3f} A")
    return reasons, {"length": len(residues),
                     "min_adjacent_CA_A": round(float(neighbor_dist.min()), 3),
                     "max_adjacent_CA_A": round(float(neighbor_dist.max()), 3),
                     "min_nonlocal_CA_A": None if shortest_nonlocal is None else round(shortest_nonlocal, 3)}


def screen(report, pdb, max_rmsd, min_cluster, min_ca, max_ca, min_nonlocal):
    reasons = []
    anchors = report.get("coordinating_cysteine_positions_in_generated_backbone", {})
    mapped = [(x.get("chain"), x.get("pdb_number")) for x in anchors.values()]
    if len(anchors) != 4 or len(set(mapped)) != 4:
        reasons.append("expected four unique coordinating-cysteine positions")
    rmsd = report.get("motif_backbone_rmsd_A")
    if rmsd is None or not np.isfinite(float(rmsd)) or float(rmsd) > max_rmsd:
        reasons.append(f"motif RMSD absent or above {max_rmsd} A")
    clash = report.get("minimum_new_backbone_to_cluster_atom_A")
    if clash is None or not np.isfinite(float(clash)) or float(clash) < min_cluster:
        reasons.append(f"new-backbone-to-reference-cluster distance absent or below {min_cluster} A")
    extra, metrics = backbone_checks(pdb, min_ca, max_ca, min_nonlocal)
    reasons.extend(extra)
    return reasons, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design-dir", type=Path, required=True)
    p.add_argument("--evaluations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-motif-rmsd", type=float, default=1.0)
    p.add_argument("--min-core-separation", type=float, default=2.0)
    p.add_argument("--min-adjacent-ca", type=float, default=3.0)
    p.add_argument("--max-adjacent-ca", type=float, default=4.5)
    p.add_argument("--min-nonlocal-ca", type=float, default=3.0)
    args = p.parse_args()
    (args.out / "accepted_pdbs").mkdir(parents=True, exist_ok=True)
    (args.out / "accepted_evaluations").mkdir(parents=True, exist_ok=True)
    rows = []
    for ev in sorted(args.evaluations.glob("design_*_evaluation.json")):
        name = ev.name.removesuffix("_evaluation.json")
        pdb = args.design_dir / f"{name}.pdb"
        reasons, measurements = [], {}
        if not pdb.exists():
            reasons.append("matching original design PDB missing")
        else:
            try:
                report = json.loads(ev.read_text())
                reasons, measurements = screen(report, pdb,
                    args.max_motif_rmsd, args.min_core_separation,
                    args.min_adjacent_ca, args.max_adjacent_ca, args.min_nonlocal_ca)
            except Exception as e:
                reasons.append(f"evaluation failed: {type(e).__name__}: {e}")
        passed = not reasons
        rows.append({"design": name, "accepted": passed, "reasons": "; ".join(reasons), **measurements})
        if passed:
            shutil.copy2(pdb, args.out / "accepted_pdbs" / pdb.name)
            shutil.copy2(ev, args.out / "accepted_evaluations" / ev.name)
            trb = pdb.with_suffix(".trb")
            if trb.exists():
                shutil.copy2(trb, args.out / "accepted_pdbs" / trb.name)
        print(f"{'ACCEPT' if passed else 'REVIEW/REJECT'} {name}: {', '.join(reasons) if reasons else 'passed coarse checks'}")
    if not rows:
        raise SystemExit("No evaluation reports found")
    with (args.out / "screening.csv").open("w", newline="") as f:
        fields = ["design", "accepted", "reasons", "length", "min_adjacent_CA_A", "max_adjacent_CA_A", "min_nonlocal_CA_A"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Coarse screening only. Visually inspect accepted structures before MPNN.")
    print(f"{sum(r['accepted'] for r in rows)}/{len(rows)} accepted -> {args.out}")

if __name__ == "__main__":
    main()
