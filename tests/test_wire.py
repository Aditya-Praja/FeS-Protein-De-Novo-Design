"""Synthetic geometrical fixtures exercise interfaces, never claim model success."""

from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from fes_wire import geometry, symmetry, motif, ranking, cofactor_network, cli
from fes_wire.config import DEFAULT, load_config
from fes_wire.structures import read, write, coords, cofactors, empty, transformed
from fes_wire.state import dump, load, stage

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    result = deepcopy(DEFAULT)
    for k in ("source", "motif", "metadata", "context"):
        result[k] = str(ROOT / result[k])
    return result


@pytest.mark.parametrize("order", [3, 4, 5, 6, 10])
def test_cyclic_preserves_geometry(cfg, order):
    cfg["order"] = order
    cfg["radius_A"] = 40
    ring, _ = symmetry.prepare(cfg)
    assert len(cofactors(ring)) == order
    first = coords(next(ring.get_chains()).get_atoms())
    for i, c in enumerate(ring.get_chains()):
        assert np.allclose(
            coords(c.get_atoms()), first @ geometry.z_rotation(i * 360 / order).T, atol=1e-5
        )


def test_fit():
    x = np.array([[1, 2, 3], [4, 2, 1], [0, 1, 3], [4, 5, 6.0]])
    rotation = geometry.z_rotation(67)
    y = x @ rotation.T + [3, 4, 5]
    r, t, error = geometry.fit(x, y)
    assert error < 1e-10
    assert np.allclose(r, rotation)
    assert np.allclose(t, [3, 4, 5])
    with pytest.raises(ValueError):
        geometry.fit(np.zeros((4, 3)), np.zeros((4, 3)))


def test_motif_and_malformed(cfg, tmp_path):
    valid = motif.validate(cfg["motif"], cfg["source"], cfg["metadata"])
    assert len(valid["donors"]) == 4
    broken = read(cfg["motif"])
    next(broken.get_residues()).detach_child("SG")
    write(broken, tmp_path / "broken.pdb")
    with pytest.raises(ValueError):
        motif.validate(tmp_path / "broken.pdb", cfg["source"], cfg["metadata"])
    (tmp_path / "empty.pdb").write_text("END\n")
    with pytest.raises(ValueError):
        read(tmp_path / "empty.pdb")


def test_roundtrip_and_stack(cfg, tmp_path):
    ring, _ = symmetry.prepare(cfg)
    write(ring, tmp_path / "ring.cif")
    assert np.allclose(
        coords(ring.get_atoms()), coords(read(tmp_path / "ring.cif").get_atoms()), atol=0.001
    )
    assembly, mapping = symmetry.stack(ring, 3, 25, 30)
    assert len(mapping) == 12
    first = coords(next(ring.get_chains()).get_atoms())
    final = coords(list(assembly.get_chains())[8].get_atoms())
    assert np.allclose(final, first @ geometry.z_rotation(60).T + [0, 0, 50])
    graph = cofactor_network.analyze(assembly, mapping, 100)
    assert graph["spans_layers"]
    assert len(graph["components"]) == 1
    assert geometry.unique_twists([0, 90, 180, 45], 4) == [0, 45]


def test_contacts_ranking_config(tmp_path):
    assert geometry.contacts(np.array([[0, 0, 0]]), np.array([[1, 0, 0]]), 2) == (1, 1.0)
    rows = [
        dict(
            candidate="bad",
            reasons=["clash"],
            axial_contacts=100,
            motif_rmsd_A=0,
            network_spans_layers=True,
        ),
        dict(
            candidate="good",
            reasons=[],
            axial_contacts=1,
            motif_rmsd_A=0,
            network_spans_layers=True,
        ),
    ]
    ranked = ranking.rank(rows, DEFAULT)
    assert ranked[0]["candidate"] == "good"
    assert ranked[1]["score"] is None
    (tmp_path / "bad.yaml").write_text("order: 11")
    with pytest.raises(ValueError):
        load_config(tmp_path / "bad.yaml")


def test_cache(tmp_path):
    source = tmp_path / "input"
    source.write_text("a")
    with stage(tmp_path, "test", [source], {}) as (folder, cached):
        assert not cached
        (folder / "result").write_text("ok")
    with stage(tmp_path, "test", [source], {}) as (_, cached):
        assert cached
    (folder / "result").write_text("altered")
    with stage(tmp_path, "test", [source], {}) as (_, cached):
        assert not cached
    source.write_text("b")
    with stage(tmp_path, "test", [source], {}) as (other, cached):
        assert not cached and folder != other


def test_synthetic_downstream(cfg, tmp_path, monkeypatch):
    """Full source monomer rigid copies, explicitly synthetic metadata in temp storage."""
    monkeypatch.chdir(tmp_path)
    cfg.update(
        radius_A=80,
        clash_A=0.1,
        min_ring_contacts=0,
        min_axial_contacts=0,
        rises_A=[60],
        twists_deg=[0],
        top_k=1,
    )
    run = Path("runs") / "synthetic_test"
    cfg["run_id"] = "synthetic_test"
    cli.validate(cfg, run, False)
    prepared = cli.prepare(cfg, run, False)
    transforms = load(prepared / "transforms.json")["transforms"]
    source = read(cfg["source"])
    base = empty()
    chain = deepcopy(source[0]["A"])
    for r in list(chain):
        if r.id[0] != " " and r.resname != "FES":
            chain.detach_child(r.id)
    base[0].add(chain)
    ring = empty()
    mapping = {}
    validation = motif.validate(cfg["motif"], cfg["source"], cfg["metadata"])
    for t in transforms:
        copy = transformed(base, np.array(t["rotation"]), np.array(t["translation"]))
        chain = deepcopy(next(copy.get_chains()))
        chain.id = t["chain"]
        ring[0].add(chain)
        for key in validation["atom_selections"]:
            mapping[chain.id + key[1:]] = chain.id + key[1:]
    folder = run / "rfd_outputs" / "synthetic_fixture"
    folder.mkdir(parents=True)
    write(ring, folder / "synthetic.cif")
    dump(
        folder / "synthetic.json",
        dict(
            synthetic_fixture=True,
            diffused_index_map=mapping,
            specification={"symmetry": {"id": "C4"}},
            metrics={},
        ),
    )
    processed = cli.process(cfg, run, False)
    assert len(load(processed / "designs.json")) == 1
    cli.rings(cfg, run, False)
    cli.stacks(cfg, run, False)
    ranked = cli.rank(cfg, run, False)
    assert load(ranked / "selected.json"), load(ranked / "ranked.json")
    output = cli.mpnn(cfg, run, False)
    exports = load(output / "candidates.json")
    assert len(exports) == 1
    monkeypatch.setattr("sys.argv", ["fes-wire", "compare", "--run-ids", "synthetic_test"])
    cli.main()
    assert list(Path("outputs/wire_comparisons/comparison").glob("*/comparison.csv"))
    constraints = load(Path(exports[0]["stack"]).parent / "constraints.json")
    assert len(constraints["fixed_residues"]) == 48
    assert all(len(g) == 12 for g in constraints["equivalent_positions"])
    from fes_wire.rfd3 import normalize

    without_cofactor = ring.copy()
    for cluster in list(cofactors(without_cofactor)):
        cluster.parent.detach_child(cluster.id)
    write(without_cofactor, tmp_path / "reconstruction.cif")
    dump(tmp_path / "reconstruction.json", load(folder / "synthetic.json"))
    reconstructed = normalize(
        tmp_path / "reconstruction.cif",
        read(prepared / "symmetric_motif.pdb"),
        validation,
        cfg,
        tmp_path / "reconstructed.cif",
    )
    assert reconstructed["cofactor_reconstructed"]
    assert len(cofactors(read(tmp_path / "reconstructed.cif"))) == 4
    broken = ring.copy()
    broken[0]["A"][(" ", 40, " ")].detach_child("SG")
    write(broken, tmp_path / "missing_sg.cif")
    dump(tmp_path / "missing_sg.json", load(folder / "synthetic.json"))
    with pytest.raises(ValueError, match="Constrained atom missing"):
        normalize(
            tmp_path / "missing_sg.cif",
            read(prepared / "symmetric_motif.pdb"),
            validation,
            cfg,
            tmp_path / "invalid.cif",
        )
    data = load(folder / "synthetic.json")
    del data["diffused_index_map"]["A40"]
    dump(folder / "synthetic.json", data)
    rejected = cli.process(cfg, run, False)
    assert not load(rejected / "designs.json")
    assert "mapping" in load(rejected / "rejected.json")[0]["reasons"][0]


def test_gzip_duplicate_altloc_and_cofactor(cfg, tmp_path):
    import gzip

    core = read(cfg["motif"])
    write(core, tmp_path / "core.cif")
    with gzip.open(tmp_path / "core.cif.gz", "wt") as handle:
        handle.write((tmp_path / "core.cif").read_text())
    assert len(cofactors(read(tmp_path / "core.cif.gz"))) == 1
    content = Path(cfg["motif"]).read_text().splitlines()
    (tmp_path / "duplicate.pdb").write_text("\n".join([content[0]] + content) + "\n")
    with pytest.raises(Exception):
        read(tmp_path / "duplicate.pdb")
    altered = core.copy()
    next(altered.get_atoms()).set_altloc("A")
    write(altered, tmp_path / "alt.pdb")
    with pytest.raises(ValueError, match="alternate"):
        read(tmp_path / "alt.pdb")
    cofactors(core)[0].detach_child("FE1")
    with pytest.raises(ValueError, match="inventory"):
        cofactors(core)


def test_cache_force_failure_and_lock(tmp_path):
    source = tmp_path / "source"
    source.write_text("source")
    with pytest.raises(RuntimeError, match="deliberate"):
        with stage(tmp_path, "retry", [source], {}) as (folder, cached):
            (folder / "partial").write_text("retained")
            raise RuntimeError("deliberate interruption")
    assert load(folder / "stage.json")["status"] == "failed"
    with stage(tmp_path, "retry", [source], {}) as (_, cached):
        assert not cached
    with stage(tmp_path, "retry", [source], {}, force=True) as (_, cached):
        assert not cached
        with pytest.raises(RuntimeError, match="locked"):
            with stage(tmp_path, "retry", [source], {}):
                pass


def test_legacy_backbone_parsing():
    paths = sorted((ROOT / "outputs/rfdiffusion").rglob("*.pdb"))
    if not paths:
        pytest.skip("Historical datasets are optional on clean installations")
    structure = read(paths[0])
    assert len(list(structure.get_atoms())) > 100


def test_legacy_length_parser_rejects_partial_nonnumeric():
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "legacy_motif", ROOT / "scripts/prepare_rfdiffusion_motif.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_length_range("20-bad")
