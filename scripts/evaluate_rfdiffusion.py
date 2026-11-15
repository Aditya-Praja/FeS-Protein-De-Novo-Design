from __future__ import annotations
import argparse
import csv
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
from Bio.PDB import Atom, Chain, PDBIO, PDBParser, Residue

BACKBONE = ('N', 'CA', 'C', 'O')


def kabsch_fit(reference: np.ndarray, generated: np.ndarray):
    """Return row-vector R, translation t: reference @ R + t ~ generated.

    A proper rotation det(R)=+1 is enforced (no reflection).
    """
    p = np.asarray(reference, dtype=float)
    q = np.asarray(generated, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
        raise ValueError('Expected matching coordinate arrays with >=3 3D points')
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    u, _, vt = np.linalg.svd((p-pc).T @ (q-qc))
    sign = np.linalg.det(u @ vt)
    r = u @ np.diag([1.0, 1.0, np.sign(sign)]) @ vt
    t = qc - pc @ r
    return r, t


def get_residue(model, chain_and_resid):
    chain_id, number = chain_and_resid
    if isinstance(chain_id, bytes):
        chain_id = chain_id.decode()
    return model[str(chain_id)][(' ', int(number), ' ')]


def load_trb(path: Path):
    
    with path.open('rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f'Invalid .trb metadata: {path}')
    for key in ('con_ref_pdb_idx', 'con_hal_pdb_idx'):
        if key not in obj:
            raise ValueError(f'{path} missing {key}; check RFdiffusion version/output')
    return obj


def collect_superposition_atoms(ref_model, new_model, ref_idx, new_idx):
    ps, qs = [], []
    if len(ref_idx) != len(new_idx):
        raise ValueError('TRB reference and generated index arrays have different lengths')
    pairs = []
    for a, b in zip(ref_idx, new_idx):
        original = get_residue(ref_model, a)
        generated = get_residue(new_model, b)
        pairs.append((a, b))
        for name in BACKBONE:
            if name in original and name in generated:
                ps.append(np.asarray(original[name].coord, dtype=float))
                qs.append(np.asarray(generated[name].coord, dtype=float))
    if len(ps) < 3:
        raise ValueError('Not enough backbone atoms mapped by .trb')
    return np.stack(ps), np.stack(qs), pairs


def build_combined_pdb(design_structure, metal_atoms, output_path):
    combined = deepcopy(design_structure)
    model = combined[0]
    chain_id = next((c for c in 'ZYXWV' if c not in model), None)
    if chain_id is None:
        raise ValueError('No free chain ID available for cluster visualization')
    chain = Chain.Chain(chain_id)
    model.add(chain)
    residue = Residue.Residue(('H_FES', 901, ' '), 'FES', ' ')
    chain.add(residue)
    for i, item in enumerate(metal_atoms, 1):
        element = item['element'].strip().upper()
        name = item['name'].strip()
        coords = np.asarray(item['aligned_coordinates'], dtype=np.float32)
        atom = Atom.Atom(name, coords, 20.0, 1.0, ' ', f'{name:>4}', i, element=element)
        residue.add(atom)
    io = PDBIO()
    io.set_structure(combined)
    io.save(str(output_path))


def calculate_metrics(reference_path, metadata, design_path, trb_path, output_dir):
    parser = PDBParser(QUIET=True)
    source = parser.get_structure('source', str(reference_path))[0]
    full_design = parser.get_structure('candidate', str(design_path))
    generated = full_design[0]
    trb = load_trb(trb_path)
    p, q, pairs = collect_superposition_atoms(
        source, generated, trb['con_ref_pdb_idx'], trb['con_hal_pdb_idx'])
    r, t = kabsch_fit(p, q)
    fitted = p @ r + t
    motif_rmsd = float(np.sqrt(np.mean(np.sum((fitted-q)**2, axis=1))))

    # Transfer the original Fe2S2 atoms with the SAME rigid rotation/translation.
    metal_atoms = []
    for atom in metadata['cluster_atoms']:
        record = dict(atom)
        record['aligned_coordinates'] = (np.asarray(atom['coordinates']) @ r + t).tolist()
        metal_atoms.append(record)
    core = np.stack([np.asarray(a['aligned_coordinates']) for a in metal_atoms])

    # Exclude fixed motif residues: cysteine-to-Fe distances are a distinct problem.
    motif_out_ids = {(str(b[0]), int(b[1])) for _, b in pairs}
    other_atoms = []
    for residue in generated.get_residues():
        loc = (residue.get_parent().id, int(residue.id[1]))
        if loc in motif_out_ids or residue.id[0] != ' ':
            continue
        for name in BACKBONE:
            if name in residue:
                other_atoms.append(np.asarray(residue[name].coord, dtype=float))
    if other_atoms:
        x = np.stack(other_atoms)
        min_separation = float(np.min(np.linalg.norm(x[:,None,:] - core[None,:,:],axis=2)))
    else:
        min_separation = None

    # Map the four original cysteine residue identities to generated chain positions.
    mapping = {(str(a[0]), int(a[1])): (str(b[0]), int(b[1])) for a, b in pairs}
    anchor_out = {}
    for ligand in metadata['coordinating_residues']:
        source_key = (str(ligand['chain']), int(ligand['number']))
        if source_key not in mapping:
            raise ValueError(f'Cysteine anchor absent from .trb mapping: {source_key}')
        dest = mapping[source_key]
        anchor_out[f'{source_key[0]}:{source_key[1]}'] = {'chain': dest[0], 'pdb_number': dest[1]}

    # ProteinMPNN uses 1-indexed sequence positions, not PDB residue numbers.
    ordinal_by_chain = {}
    for chain in generated:
        ordinal_by_chain[chain.id] = {int(res.id[1]): i for i, res in enumerate(
            [r for r in chain if r.id[0] == ' '], 1)}
    for anchor in anchor_out.values():
        anchor['sequence_ordinal_1based'] = ordinal_by_chain[anchor['chain']][anchor['pdb_number']]

    generated_cluster = output_dir / f'{design_path.stem}_plus_reference_FES.pdb'
    build_combined_pdb(full_design, metal_atoms, generated_cluster)
    report = {
        'design': design_path.name,
        'mapped_motif_backbone_atoms': len(p),
        'motif_backbone_rmsd_A': round(motif_rmsd, 4),
        'minimum_new_backbone_to_cluster_atom_A': None if min_separation is None else round(min_separation, 4),
        'possible_backbone_clash_if_below_2p0_A': None if min_separation is None else min_separation < 2.0,
        'coordinating_cysteine_positions_in_generated_backbone': anchor_out,
        'metal_atoms_reference_overlay': metal_atoms,
        'combined_pdb': str(generated_cluster),
        'warning': 'Cluster positions are superposed from experimental reference, not predicted or relaxed. '
                   'New designs have no trustworthy SG rotamers or proven Fe-S binding chemistry.'
    }
    (output_dir / f'{design_path.stem}_evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-pdb', type=Path, required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--design-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text())
    pdbs = sorted(args.design_dir.glob('design_*.pdb'))
    
    pdbs = [p for p in pdbs if '_plus_reference_FES' not in p.stem]
    if not pdbs:
        raise SystemExit(f'No design_*.pdb in {args.design_dir}')
    reports = []
    for pdb in pdbs:
        trb = pdb.with_suffix('.trb')
        if not trb.is_file():
            print(f'SKIP {pdb.name}: no matching {trb.name}')
            continue
        try:
            rep = calculate_metrics(args.reference_pdb, metadata, pdb, trb, args.out)
            reports.append(rep)
            print(f"{pdb.name}: motif RMSD={rep['motif_backbone_rmsd_A']:.3f} Å; "
                  f"min new-backbone-to-core={rep['minimum_new_backbone_to_cluster_atom_A']} Å")
        except Exception as exc:
            print(f'SKIP {pdb.name}: {type(exc).__name__}: {exc}')
    if not reports:
        raise SystemExit('No designs evaluated successfully. Inspect .trb and PDB files.')
    with (args.out/'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['design','motif_backbone_rmsd_A',
               'minimum_new_backbone_to_cluster_atom_A','possible_backbone_clash_if_below_2p0_A'])
        w.writeheader()
        for r in reports:
            w.writerow({k:r.get(k) for k in w.fieldnames})
    print(f'Saved {len(reports)} evaluations to {args.out}/')

if __name__ == '__main__':
    main()
