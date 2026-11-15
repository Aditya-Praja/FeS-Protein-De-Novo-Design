import argparse
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

BACKBONE = ("N", "CA", "C", "O")


def check_one(pdb_path: Path, fixed_by_chain, parsed=None):
    model = PDBParser(QUIET=True).get_structure("candidate", str(pdb_path))[0]
    seen = set()
    selected_count = 0
    for chain in model:
        residues = [r for r in chain if r.id[0] == " "]
        if not residues:
            continue
        seen.add(chain.id)
        nums = [r.id[1] for r in residues]
        if any(r.id[2].strip() for r in residues):
            raise ValueError("insertion codes unsupported; renumber with explicit mapping")
        if nums != list(range(nums[0], nums[0] + len(nums))):
            raise ValueError("residue-number gaps: ProteinMPNN parser pads missing numbers; mapping needs revision")
        for r in residues:
            if any(a not in r for a in BACKBONE):
                raise ValueError(f"missing backbone atom at {chain.id}:{r.id[1]}")
            if not all(np.isfinite(r[a].coord).all() for a in BACKBONE):
                raise ValueError(f"non-finite coordinates at {chain.id}:{r.id[1]}")
        locked = fixed_by_chain.get(chain.id, [])
        for ordinal in locked:
            if type(ordinal) is not int or not 1 <= ordinal <= len(residues):
                raise ValueError(f"fixed position {ordinal} outside chain {chain.id}")
            if residues[ordinal-1].resname != "CYS":
                raise ValueError(f"position {chain.id}:{ordinal} is {residues[ordinal-1].resname}, not CYS")
        selected_count += len(locked)
        if parsed is not None:
            seq = parsed.get(f"seq_chain_{chain.id}")
            if seq is None or len(seq) != len(residues):
                raise ValueError(f"ProteinMPNN parser sequence length mismatch in chain {chain.id}")
            actual_seq = ''.join(seq1(r.resname, custom_map={"UNK":"X"}) for r in residues)
            if actual_seq != seq:
                raise ValueError(f"ProteinMPNN parsed sequence mismatch in chain {chain.id}")
            if any(seq[i-1] != "C" for i in locked):
                raise ValueError(f"ProteinMPNN parsed anchors not CYS in chain {chain.id}")
    if seen != set(fixed_by_chain):
        raise ValueError(f"chain set mismatch: PDB {seen}, fixed {set(fixed_by_chain)}")
    if selected_count != 4:
        raise ValueError(f"expected four fixed cysteines, found {selected_count}")
    return selected_count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb-dir", type=Path, required=True)
    ap.add_argument("--fixed", type=Path, required=True)
    ap.add_argument("--parsed", type=Path, help="Optional official parse_multiple_chains.py JSONL")
    args = ap.parse_args()
    # Companion prepare_mpnn_inputs.py writes one JSONL line containing all designs.
    fixed = {}
    for line in args.fixed.read_text().splitlines():
        if line.strip():
            fixed.update(json.loads(line))
    parsed = {}
    if args.parsed is not None:
        for line in args.parsed.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                parsed[entry["name"]] = entry
    found = {p.stem for p in args.pdb_dir.glob("*.pdb")}
    if found != set(fixed):
        raise SystemExit(f"PDB names {sorted(found)} do not match fixed JSONL names {sorted(fixed)}")
    if args.parsed is not None and found != set(parsed):
        raise SystemExit("Parsed JSONL design names do not match input PDB names")
    for name in sorted(found):
        n = check_one(args.pdb_dir / f"{name}.pdb", fixed[name], parsed.get(name) if args.parsed else None)
        print(f"OK {name}: {n} fixed cysteines verified")
    print(f"Verified {len(found)} ProteinMPNN input PDB(s)")

if __name__ == "__main__":
    main()
