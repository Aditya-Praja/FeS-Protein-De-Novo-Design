import argparse
import json
from pathlib import Path
from Bio.PDB import PDBIO, PDBParser


def prepare_design(pdb_path, evaluation_path, target_dir):
    record = json.loads(evaluation_path.read_text())
    structure = PDBParser(QUIET=True).get_structure('mpnn', str(pdb_path))
    model = structure[0]
    pos_by_chain = {}
    for original_anchor, mapped in record['coordinating_cysteine_positions_in_generated_backbone'].items():
        ch = mapped['chain']
        num = int(mapped['pdb_number'])
        residue = model[ch][(' ', num, ' ')]
        # No new atoms: this label conveys *sequence identity*, not a resolved rotamer.
        residue.resname = 'CYS'
        pos_by_chain.setdefault(ch, []).append(int(mapped['sequence_ordinal_1based']))
    if sum(len(x) for x in pos_by_chain.values()) != 4:
        raise ValueError(f'{pdb_path.name}: expected exactly 4 coordinating Cys')
    for ch in pos_by_chain:
        pos_by_chain[ch] = sorted(pos_by_chain[ch])
    io = PDBIO()
    io.set_structure(structure)
    out_path = target_dir / pdb_path.name
    io.save(str(out_path))
    return pdb_path.stem, pos_by_chain


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--design-dir', type=Path, required=True)
    ap.add_argument('--evaluations', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    pdb_folder = args.out / 'inputs'
    pdb_folder.mkdir(parents=True, exist_ok=True)
    fixed = {}
    for ev in sorted(args.evaluations.glob('design_*_evaluation.json')):
        stem = ev.name.removesuffix('_evaluation.json')
        original = args.design_dir / f'{stem}.pdb'
        if not original.exists():
            raise FileNotFoundError(f'Could not find {original}')
        name, positions = prepare_design(original, ev, pdb_folder)
        fixed[name] = positions
        print(f'{name}: lock coordinating cysteines {positions}')
    if not fixed:
        raise SystemExit('No evaluation JSONs found; evaluate RFdiffusion PDBs first')
    target = args.out/'fixed_positions.jsonl'
    target.write_text(json.dumps(fixed)+'\n')
    print(f'Wrote {target}')

if __name__ == '__main__':
    main()
