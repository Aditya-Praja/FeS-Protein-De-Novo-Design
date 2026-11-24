from pathlib import Path
import re
import numpy as np
import yaml

DEFAULT = dict(
    source="data/raw_pdb/1off.pdb",
    motif="data/motifs/1off_2fe2s/motif.pdb",
    metadata="data/motifs/1off_2fe2s/metadata.json",
    context="data/motifs/1off_2fe2s_f2/rfdiffusion_input.pdb",
    run_id="c4_pilot",
    order=4,
    radius_A=18.0,
    orientation_deg=[0, 0, 0],
    length=100,
    designs=2,
    seed=42,
    rises_A=[20.0, 25.0, 30.0],
    twists_deg=[0, 15, 30, 45],
    layers=3,
    clash_A=1.5,
    contact_A=5.0,
    network_edge_A=14.0,
    max_motif_rmsd_A=1.0,
    max_symmetry_rmsd_A=1.0,
    min_axial_contacts=1,
    min_ring_contacts=1,
    require_network_path=False,
    top_k=5,
    weights={"axial_contacts": 1.0, "motif_rmsd_A": -10.0},
)


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - set(DEFAULT)
    if unknown:
        raise ValueError(f"Unknown configuration fields {sorted(unknown)}")
    cfg = dict(DEFAULT, **raw)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg["run_id"]):
        raise ValueError("run_id must contain only letters, numbers, underscore and hyphen")
    for key, low, high in (
        ("order", 2, 10),
        ("length", 10, 2000),
        ("designs", 1, 100000),
        ("layers", 2, 100),
        ("top_k", 1, 10000),
        ("seed", 0, 2**31 - 1),
    ):
        value = cfg[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    for key in (
        "radius_A",
        "clash_A",
        "contact_A",
        "network_edge_A",
        "max_motif_rmsd_A",
        "max_symmetry_rmsd_A",
    ):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("min_axial_contacts", "min_ring_contacts"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("rises_A", "twists_deg", "orientation_deg"):
        if not cfg[key] or not np.isfinite(cfg[key]).all():
            raise ValueError(f"Invalid {key}")
    if len(cfg["orientation_deg"]) != 3 or min(cfg["rises_A"]) <= 0:
        raise ValueError("Need three Euler angles and positive rises")
    if set(cfg["weights"]) - {
        "axial_contacts",
        "motif_rmsd_A",
        "symmetry_rmsd_A",
        "network_components",
    }:
        raise ValueError("Unsupported ranking metric")
    if not all(np.isfinite(v) for v in cfg["weights"].values()):
        raise ValueError("Nonfinite ranking weight")
    return cfg
