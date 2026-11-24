"""Unified stages. Model execution remains an explicit, separate user action."""

import argparse
import csv
import json
import logging
from pathlib import Path
from copy import deepcopy
from Bio.PDB.PDBExceptions import PDBException
from . import motif, symmetry, rfd3, evaluation, cofactor_network, ranking, ligandmpnn_inputs
from .config import load_config
from .state import stage, dump, load, latest, digest
from .structures import read, write, empty
from .geometry import unique_twists

LOG = logging.getLogger(__name__)


def table(path, rows):
    if not rows:
        Path(path).write_text("candidate,passed,reasons\n")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            for row in rows
        )


def validate(cfg, run, force):
    with stage(
        run, "motif_validation", [cfg[k] for k in ("motif", "source", "metadata")], cfg, force
    ) as (out, cached):
        if not cached:
            result = motif.validate(cfg["motif"], cfg["source"], cfg["metadata"])
            dump(out / "validation.json", result)
            (out / "report.md").write_text(
                "# Experimental motif validation\n\nSource atom coordinates and identities match.\n\n"
                + "\n".join(
                    f"- {d['residue']} {d['name']} {d['atom']} nearest Fe {min(d['distances_A']):.3f} Å"
                    for d in result["donors"]
                )
                + "\n"
            )
    return out


def prepare(cfg, run, force):
    validated = latest(run, "motif_validation") / "validation.json"
    if (run / "manifest.json").exists() and rfd3.discover(run / "rfd_outputs"):
        old_manifest = load(run / "manifest.json")
        old = old_manifest["config"]
        original_stage = load(Path(old_manifest["inputs"]) / "stage.json")
        for path, expected_hash in original_stage["inputs"].items():
            if not Path(path).is_file() or digest(path) != expected_hash:
                raise ValueError("Generation input changed after inference. Use a new run_id")
        generation_keys = (
            "source",
            "motif",
            "metadata",
            "context",
            "order",
            "radius_A",
            "orientation_deg",
            "length",
            "designs",
            "seed",
        )
        if any(old[k] != cfg[k] for k in generation_keys):
            raise ValueError("Generation configuration changed after inference. Use a new run_id")
    with stage(
        run, "rfd_inputs", [validated, cfg["context"], cfg["motif"], cfg["source"]], cfg, force
    ) as (out, cached):
        if not cached:
            structure, transforms = symmetry.prepare(cfg)
            write(structure, out / "symmetric_motif.pdb")
            write(structure, out / "symmetric_motif.cif")
            spec = rfd3.specification(cfg, load(validated), out / "symmetric_motif.pdb")
            dump(out / "inputs.json", spec)
            dump(
                out / "transforms.json",
                dict(axis=[0, 0, 1], origin=[0, 0, 0], transforms=transforms),
            )
            dump(
                out / "manifest.json",
                dict(
                    config=cfg,
                    model_revision=rfd3.REVISION,
                    input_sha256=digest(out / "symmetric_motif.pdb"),
                    expected_designs=cfg["designs"],
                    outputs=str(run / "rfd_outputs"),
                    integration_status="Awaiting real symmetric FES pilot",
                ),
            )
            jobs = []
            for index in range(cfg["designs"]):
                jobs.append(
                    dict(
                        index=index,
                        seed=cfg["seed"] + index,
                        inputs=str(out / "inputs.json"),
                        out_dir=str(run / "rfd_outputs" / f"job_{index:05d}"),
                    )
                )
            dump(out / "jobs.json", jobs)
    dump(run / "manifest.json", dict(config=cfg, inputs=str(out), rfd3_revision=rfd3.REVISION))
    return out


def process(cfg, run, force):
    prepared = latest(run, "rfd_inputs")
    validated = latest(run, "motif_validation") / "validation.json"
    paths = rfd3.discover(run / "rfd_outputs")
    if not paths:
        raise ValueError(f"No outputs in {run / 'rfd_outputs'}. Run RFD3 manually first")
    inputs = paths + [
        p.with_name(p.name.removesuffix(".gz").rsplit(".", 1)[0] + ".json") for p in paths
    ]
    existing = [p for p in inputs if p.exists()]
    with stage(
        run, "processed", existing + [prepared / "symmetric_motif.pdb", validated], cfg, force
    ) as (out, cached):
        if not cached:
            reference = read(prepared / "symmetric_motif.pdb")
            good, bad = [], []
            for path in paths:
                identifier = (
                    digest(path)[:12] + "_" + path.parent.name + "_" + path.name.split(".")[0]
                )
                try:
                    sidecar = path.with_name(
                        path.name.removesuffix(".gz").rsplit(".", 1)[0] + ".json"
                    )
                    with stage(
                        run / "design_cache",
                        identifier,
                        [path, sidecar, validated, prepared / "symmetric_motif.pdb"],
                        cfg,
                        force,
                    ) as (design_folder, design_cached):
                        if not design_cached:
                            record = rfd3.normalize(
                                path,
                                reference,
                                load(validated),
                                cfg,
                                design_folder / "structure.cif",
                            )
                            record["design"] = identifier
                            dump(design_folder / "record.json", record)
                        record = load(design_folder / "record.json")
                    good.append(record)
                except (ValueError, KeyError, OSError, EOFError, TypeError, PDBException) as exc:
                    bad.append(dict(source=str(path), reasons=[str(exc)]))
            dump(out / "designs.json", good)
            dump(out / "rejected.json", bad)
            dump(
                out / "completion.json",
                dict(
                    expected_designs=cfg["designs"],
                    discovered=len(paths),
                    normalized=len(good),
                    rejected=len(bad),
                    expected_count_reached=len(paths) == cfg["designs"],
                ),
            )
            LOG.info("Normalized %s designs, rejected %s", len(good), len(bad))
    return out


def rings(cfg, run, force):
    processed = latest(run, "processed")
    records = load(processed / "designs.json")
    with stage(
        run,
        "ring_evaluation",
        [processed / "designs.json"] + [r["structure"] for r in records],
        cfg,
        force,
    ) as (out, cached):
        if not cached:
            rows = []
            for record in records:
                with stage(
                    run / "ring_cache",
                    record["design"],
                    [record["structure"]],
                    dict(config=cfg, record=record),
                    force,
                ) as (ring_folder, ring_cached):
                    if not ring_cached:
                        row = evaluation.evaluate_ring(read(record["structure"]), record, cfg)
                        dump(ring_folder / "metrics.json", row)
                    rows.append(load(ring_folder / "metrics.json"))
            dump(out / "rings.json", rows)
            table(out / "rings.csv", rows)
    return out


def stacks(cfg, run, force):
    processed, ringdir = latest(run, "processed"), latest(run, "ring_evaluation")
    records = load(processed / "designs.json")
    ringrows = {r["design"]: r for r in load(ringdir / "rings.json")}
    with stage(
        run,
        "stack_evaluation",
        [processed / "designs.json", ringdir / "rings.json"] + [r["structure"] for r in records],
        cfg,
        force,
    ) as (out, cached):
        if not cached:
            rows = []
            for record in records:
                ring = read(record["structure"])
                for rise in cfg["rises_A"]:
                    for twist in unique_twists(cfg["twists_deg"], cfg["order"]):
                        identifier = f"{record['design']}_r{rise:g}_t{twist:g}"
                        candidate_config = dict(
                            config=cfg,
                            record=record,
                            ring=ringrows[record["design"]],
                            rise=rise,
                            twist=twist,
                        )
                        with stage(
                            run / "stack_cache",
                            identifier,
                            [record["structure"]],
                            candidate_config,
                            force,
                        ) as (candidate_folder, candidate_cached):
                            if not candidate_cached:
                                assembly, mapping = symmetry.stack(ring, cfg["layers"], rise, twist)
                                screening = evaluation.evaluate_stack(assembly, mapping, cfg)
                                graph = cofactor_network.analyze(
                                    assembly, mapping, cfg["network_edge_A"]
                                )
                                dump(candidate_folder / "network.json", graph)
                                row = dict(
                                    candidate=identifier,
                                    design=record["design"],
                                    rise_A=rise,
                                    twist_deg=twist,
                                    layers=cfg["layers"],
                                    order=cfg["order"],
                                    reasons=ringrows[record["design"]]["reasons"]
                                    + screening["reasons"],
                                    axial_contacts=screening["axial_contacts"],
                                    layer_pairs=screening["layer_pairs"],
                                    motif_rmsd_A=record["motif_rmsd_A"],
                                    symmetry_rmsd_A=ringrows[record["design"]]["symmetry_rmsd_A"],
                                    network_components=len(graph["components"]),
                                    network_spans_layers=graph["spans_layers"],
                                    network=str(candidate_folder / "network.json"),
                                )
                                dump(candidate_folder / "metrics.json", row)
                            rows.append(load(candidate_folder / "metrics.json"))
            dump(out / "stacks.json", rows)
            table(out / "stacks.csv", rows)
    return out


def rank(cfg, run, force):
    stackdir = latest(run, "stack_evaluation")
    with stage(run, "ranked_candidates", [stackdir / "stacks.json"], cfg, force) as (out, cached):
        if not cached:
            rows = ranking.rank(load(stackdir / "stacks.json"), cfg)
            passing = [r for r in rows if r["passed"]]
            dump(out / "ranked.json", rows)
            dump(out / "selected.json", passing[: cfg["top_k"]])
            for name, subset in [
                ("all", rows),
                ("passing", passing),
                ("rejected", [r for r in rows if not r["passed"]]),
            ]:
                table(out / (name + ".csv"), subset)
            (out / "summary.md").write_text(
                f"# Geometric screening\n\n{len(passing)} of {len(rows)} assemblies pass configured criteria.\n\nThese are structural proxies. Experimental assembly and electron transfer remain untested.\n"
            )
            if rows:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 3, figsize=(12, 3), layout="constrained")
                for ax, key in zip(axes, ["motif_rmsd_A", "axial_contacts", "network_components"]):
                    ax.hist([r[key] for r in rows if r[key] is not None], bins=15)
                    ax.set_xlabel(key)
                    ax.set_ylabel("Assemblies")
                fig.savefig(out / "distributions.png", dpi=160)
                plt.close(fig)
    return out


def mpnn(cfg, run, force):
    ranked, processed = latest(run, "ranked_candidates"), latest(run, "processed")
    records = {r["design"]: r for r in load(processed / "designs.json")}
    selected = load(ranked / "selected.json")
    inputs = [ranked / "selected.json", processed / "designs.json"] + [
        records[s["design"]]["structure"] for s in selected
    ]
    with stage(run, "ligandmpnn_inputs", inputs, cfg, force) as (out, cached):
        if not cached:
            exports = []
            for selected_row in selected:
                record = records[selected_row["design"]]
                ring = read(record["structure"])
                assembly, mapping = symmetry.stack(
                    ring, selected_row["layers"], selected_row["rise_A"], selected_row["twist_deg"]
                )
                folder = out / selected_row["candidate"]
                write(assembly, folder / "assembly.cif")
                cofactor_network.plot_network(load(selected_row["network"]), folder / "network.png")
                stack_path = ligandmpnn_inputs.export(
                    assembly, mapping, record["fixed"], folder / "stack"
                )
                ringmap = {c.id: dict(layer=0, subunit=c.id) for c in ring[0]}
                ligandmpnn_inputs.export(ring, ringmap, record["fixed"], folder / "ring")
                monomer = empty()
                chain = deepcopy(next(ring.get_chains()))
                monomer[0].add(chain)
                ligandmpnn_inputs.export(
                    monomer, {chain.id: ringmap[chain.id]}, record["fixed"], folder / "monomer"
                )
                exports.append(
                    dict(
                        candidate=selected_row["candidate"],
                        stack=stack_path,
                        source=record["source"],
                    )
                )
            dump(out / "candidates.json", exports)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "audit",
            "compare",
            "validate-motif",
            "prepare-symmetry",
            "prepare-rfd",
            "pre-rfd",
            "process-rfd",
            "evaluate-rings",
            "evaluate-stacks",
            "rank",
            "prepare-mpnn",
            "post-rfd",
        ],
    )
    parser.add_argument("--config", default="configs/architectures/c4_pilot.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--run-ids", nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.stage == "audit":
        print(Path("docs/repository_audit.md").read_text())
        return
    if args.stage == "compare":
        if not args.run_ids:
            parser.error("compare requires --run-ids")
        paths = [
            latest(Path("runs") / name, "ranked_candidates") / "ranked.json"
            for name in args.run_ids
        ]
        with stage(
            Path("outputs/wire_comparisons"), "comparison", paths, args.run_ids, args.force
        ) as (out, cached):
            if not cached:
                rows = []
                for name, path in zip(args.run_ids, paths):
                    for row in load(path):
                        rows.append(dict(run_id=name, **row))
                table(out / "comparison.csv", rows)
                dump(out / "comparison.json", rows)
        LOG.info("Comparison %s", out)
        return
    cfg = load_config(args.config)
    if args.run_id:
        manifest = load(Path("runs") / args.run_id / "manifest.json")
        cfg = manifest["config"]
    run = Path("runs") / cfg["run_id"]
    functions = {
        "validate-motif": validate,
        "prepare-symmetry": prepare,
        "prepare-rfd": prepare,
        "process-rfd": process,
        "evaluate-rings": rings,
        "evaluate-stacks": stacks,
        "rank": rank,
        "prepare-mpnn": mpnn,
    }
    actions = (
        [validate, prepare]
        if args.stage == "pre-rfd"
        else [process, rings, stacks, rank, mpnn]
        if args.stage == "post-rfd"
        else [functions[args.stage]]
    )
    try:
        for action in actions:
            out = action(cfg, run, args.force)
            LOG.info("Outputs %s", out)
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"Pipeline error: {exc}\n")


if __name__ == "__main__":
    main()
