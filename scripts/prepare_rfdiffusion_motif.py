from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, Select, PDBIO

BACKBONE = ["N", "CA", "C", "O"]

def parse_length_range(value: str) -> str:
    pieces = value.split("-")
    
    if len(pieces) != 2 or any(not p.isdigit() for p in pieces):
        raise argparse.ArgumentTypeError("Invalid length range")
    low, high = map(int, pieces)
    
    if low < 1 or high < low:
        raise argparse.ArgumentTypeError("Invalid length range")
    return value

def distance(a, b) -> float:
    if hasattr(a, "get_coord"):
        a = a.get_coord()

    if hasattr(b, "get_coord"):
        b = b.get_coord()

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return float(np.linalg.norm(a - b))
    

def validate_geometry(metadata: dict) -> dict:
    
    atoms = metadata["cluster_atoms"]
    iron = [np.asarray(atom["coordinates"]) for atom in atoms if atom["element"].upper() == "FE"]
    sulfide = [np.asarray(atom["coordinates"]) for atom in atoms if atom["element"].upper() == "S"]
    
    if len(iron) != 2 or len(sulfide) != 2:
        raise ValueError("Invalid geometry")
    
    if not all(np.all(np.isfinite(coord)) for coord in iron+sulfide):
        raise ValueError("Invalid geometry")
    
    ligands = metadata["coordinating_residues"]
    if len(ligands) != 4:
        raise ValueError("Invalid geometry")
    
    seen = set()
    counts = Counter()
    cysteines = []
    warnings = []
    
    for ligand in ligands:
        identity = (ligand["chain"], ligand["num"], ligand.get("insertion_code", " "))
        
        if identity in seen:
            raise ValueError(f"Duplicate ligand: {identity}")
        seen.add(identity)
        
        sg = np.asarray(ligand["sg_coordinates"])
        if not np.all(np.isfinite(sg)):
            raise ValueError("Invalid geometry")
        
        dist = [distance(sg, fe) for fe in iron]
        nearest = int(np.argmin(dist))
        counts[nearest] += 1
        if not 1.9 <= dist[nearest] <= 2.7:
            warnings.append(f"Distance out of range: {dist[nearest]}")
        
        cysteines.append({
            "chain": ligand["chain"],
            "residue": ligand["num"],
            "insertion_code": ligand.get("insertion_code", " "),
            "nearest_fe": nearest,
            "distances_to_fe": [round(d, 2) for d in dist]
        })
        
    if [counts[0], counts[1]] != [2, 2]:
        warnings.append("Incorrect Fe coordination")
        
    fe_fe = distance(iron[0], iron[1])
    
    if not 2.4 <= fe_fe <= 3.1:
        warnings.append("Fe-Fe distance out of range")
    
    bridges = [[distance(fe, s) for s in sulfide] for fe in iron]
    for i, j in enumerate(bridges):
        for k, d in enumerate(j):
            if not 1.9 <= d <= 2.6:
                warnings.append(f"Fe{i+1}-inorganic-S{k+1} = {d} A")
    
    return {
        "Fe-Fe_distance": fe_fe,
        "Fe_bridging_S_distances": bridges,
        "cysteines": cysteines,
        "ligand_per_iron": [counts[0], counts[1]],
        "warnings": warnings
    }
    
class ProteinWindows(Select):
        
        def __init__(self, chosen_residues, chain_id):
            self.chosen_residues = {id(r) for r in chosen_residues}
            self.chain_id = chain_id
        
        def accept_model(self, model):
            return model.id == 0
        
        def accept_chain(self, chain):
            return chain.id == self.chain_id
        
        def accept_residue(self, residue):
            return id(residue) in self.chosen_residues
        
def make_windows(model, metadata: dict, flank: int):
    ligands = metadata["coordinating_residues"]
    chain_ids = {record["chain"] for record in ligands}
        
    if len(chain_ids) != 1:
        raise ValueError("Multiple chains not supported")
        
    chain_id = next(iter(chain_ids))
    if chain_id not in model:
        raise ValueError("Chain not found")
        
    residues = [r for r in model[chain_id] if r.id[0] == " "]
    if not residues:
        raise ValueError("No residues found")
    index_by_id = {r.id: i for i, r in enumerate(residues)}
        
    chosen_indices = set()
    anchors = []
        
    for record in ligands:
        resid = (" ", int(record["num"]), record.get("insertion_code", " ") or " ")
        if resid not in index_by_id:
            raise ValueError("Residue not found")
        index = index_by_id[resid]
        residue = residues[index]
        if residue.resname != "CYS":
            raise ValueError("Not a CYS")
        if "SG" not in residue:
            raise ValueError("No SG atom")
        source_sg = residue["SG"]
        reference_sg = record["sg_coordinates"]
        if distance(source_sg, reference_sg) > 0.05:
            raise ValueError("SG mismatch")
        anchors.append(int(resid[1]))
        for j in range(max(0, index-flank), min(len(residues), index+flank+1)):
            chosen_indices.add(j)
            
    chosen = [residues[i] for i in sorted(chosen_indices)]
    
    if any(r.id[2] != " " for r in chosen):
        raise ValueError("Insertion code found")
    for residues in chosen:
        missing = [name for name in BACKBONE if name not in residues]
        if missing:
            raise ValueError("Missing backbone atoms: " + ", ".join(missing))
        
    segments = []
    start = prev = chosen[0].id[1]
    for residue in chosen[1:]:
        number = residue.id[1]
        if number != prev + 1:
            segments.append((start, prev))
            start = number
        prev = number
    segments.append((start, prev))
    
    return chain_id, chosen, sorted(anchors), segments

def write_rfdiffusion_input(sturcutre, residues, chain_id, destination: Path):
    io = PDBIO()
    io.set_structure(sturcutre)
    io.save(str(destination), select=ProteinWindows(residues, chain_id))
    
def generate_contig(chain_id, segments, terminal, gap) -> str:
    pieces = [terminal]
    for k, (start, end) in enumerate(segments):
        pieces.append(f"{chain_id}{start}-{end}")
        if k != len(segments) - 1:
            pieces.append(gap)
    pieces.append(terminal)
    return "contigmap.contigs=[" + "/".join(pieces) + "]"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True, help="Original full protein PDB")
    parser.add_argument("--metadata", type=Path, required=True, help="Module 9.24 JSON")
    parser.add_argument("--out", type=Path, required=True, help="Module 9.25 output directory")
    parser.add_argument("--flank", type=int, default=2, help="Neighbors on EACH side of CYS")
    parser.add_argument("--terminal", type=parse_length_range, default="10-25")
    parser.add_argument("--gap", type=parse_length_range, default="6-18")
    args = parser.parse_args()
    if args.flank < 0:
        parser.error("--flank must be non-negative")
    if not args.pdb.is_file() or not args.metadata.is_file():
        parser.error("Both --pdb and --metadata must exist")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("source_pdb") != args.pdb.name:
        raise ValueError("Metadata and source PDB names disagree; verify the inputs")

    validation = validate_geometry(metadata)
    structure = PDBParser(QUIET=True).get_structure("reference", str(args.pdb))
    model = structure[0]
    chain_id, selected, anchor_numbers, segments = make_windows(model, metadata, args.flank)
    contig = generate_contig(chain_id, segments, args.terminal, args.gap)

    args.out.mkdir(parents=True, exist_ok=True)
    write_rfdiffusion_input(structure, selected, chain_id,
                            args.out / "rfdiffusion_input.pdb")
    report = {
        "source_pdb": str(args.pdb),
        "source_metadata": str(args.metadata),
        "coordination_geometry": validation,
        "motif_chain": chain_id,
        "cysteine_anchor_residue_numbers": anchor_numbers,
        "flank_per_side": args.flank,
        "preserved_backbone_segments": [
            {"chain": chain_id, "start": start, "end": end}
            for start, end in segments
        ],
        "rfdiffusion_contig": contig,
    }
    (args.out / "design_constraints.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("Fe-Fe distance (A):", validation["Fe-Fe_distance"])
    print("Fe-bridge S distances (A):", validation["Fe_bridging_S_distances"])
    for rec in validation["cysteines"]:
        print(f"CYS {rec['chain']}:{rec['residue']}: "
              f"Fe distances {rec['distances_to_fe']}, "
              f"nearest Fe index {rec['nearest_fe']}")
    for warning in validation["warnings"]:
        print("GEOMETRY WARNING:", warning)
    print("Preserved windows:", segments)
    print("RFdiffusion argument:", f"'{contig}'")
    print("Wrote outputs in:", args.out)


if __name__ == "__main__":
    main()

