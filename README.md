# Fe–S protein nanowire design

This repository prepares and geometrically screens proteins containing an experimentally
observed [2Fe–2S] site. Identical monomers form a cyclic ring, and repeating rise/twist
transformations stack rings into finite filament models. C4 is the initial architecture.
C3–C10 are configurable within the pinned RFD3 symmetry implementation limits.

The CPU workflow covers motif validation → symmetric input preparation → **manual RFD3**
→ output normalization → ring screening → stack screening → cofactor graphs → ranking
→ LigandMPNN structural input preparation. Sequence generation and experimental validation
remain subsequent research phases. Geometry alone cannot establish electron transfer,
conductivity, affinity, thermodynamic stability or self-assembly.

## Start here

Run from the repository root. Python 3.9–3.12 supports the pinned local environment.
The independent inference environment requires Python 3.12 or newer at the recorded revision.

```bash
python3 -m venv .venv-wire
source .venv-wire/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pip install --no-deps -e .
fes-wire audit
fes-wire pre-rfd --config configs/architectures/c4_pilot.yaml
python -m pytest -q
```

The exact transfer, Della installation, submission and retrieval procedure is in
[the numbered execution guide](docs/execution.md). Definitions, assumptions, configuration
and scientific limitations are centralized in [methods](docs/methods.md). The
[audit](docs/repository_audit.md) explains reuse and migration.

## Repository layout

```text
configs/architectures/       pilot and production YAML
configs/cluster/             configurable Della resources
src/fes_wire/                CPU preparation, geometry, adapters and CLI
scripts/cluster/            manual RFD3 runner and CPU parser validation
cluster/slurm/rfd3.sbatch    array job entry point
runs/<run_id>/              stage pointers and content-addressed results
  motif_validation/         source comparison and donor inventory
  rfd_inputs/               PDB/mmCIF, constraints, transforms, jobs
  rfd_outputs/              user-generated inference output
  processed/                canonical structures and mappings
  ring_evaluation/          ring and monomer measurements
  stack_evaluation/         rise/twist grid and cofactor graphs
  ranked_candidates/        all, rejected, passing and selected candidates
  ligandmpnn_inputs/        selected monomer, ring and stack contexts
data/                      preserved source, core and flank motifs
outputs/                   preserved historical results
scripts/                   historical RFdiffusion/ProteinMPNN workflow
tests/                     synthetic geometry and interface regression tests
docs/                      audit, methods, execution and verification notes
```

## Individual stages

```bash
fes-wire validate-motif --config configs/architectures/c4_pilot.yaml
fes-wire prepare-symmetry --config configs/architectures/c4_pilot.yaml
fes-wire prepare-rfd --config configs/architectures/c4_pilot.yaml
fes-wire process-rfd --run-id c4_pilot
fes-wire evaluate-rings --run-id c4_pilot
fes-wire evaluate-stacks --run-id c4_pilot
fes-wire rank --run-id c4_pilot
fes-wire prepare-mpnn --run-id c4_pilot
fes-wire post-rfd --run-id c4_pilot
```

`prepare-symmetry` and `prepare-rfd` share one preparation stage to avoid duplicate work.
Run motif validation first, or use `pre-rfd`. `--run-id` loads the saved run configuration.
To change analysis settings, supply a YAML with the same run ID using `--config`.
Use a new run ID for a new placement, symmetry or generation configuration.

Each stage records input/output SHA-256 hashes, configuration, code hashes, dependency
versions and completion status. Unchanged stages are reused. `--force` reruns the matching
stage. Changed inputs create a distinct result directory. Stage pointer JSON files identify
the current result. Interrupted stages retain their files and report failure. A `.lock`
prevents concurrent writes to the same stage. After a process crash, verify the process
has ended before removing its stale lock. Historical outputs are never deleted.

## Verification status

The local pre-RFD workflow and synthetic downstream integration are exercised by tests.
Synthetic fixtures are temporary algorithm tests. They do not demonstrate RFD3 inference
or a successful nanowire. The pinned model source and documented metadata were inspected.
The RFD3 CPU parser in its inference environment, FES CCD interpretation, real symmetric
output mapping and LigandMPNN's installed parser still require the checks described in
[verification notes](docs/verification.md). Neither model was executed during development.

Compare completed runs with `fes-wire compare --run-ids c3_pilot c4_pilot c5_pilot`.
The comparison table retains symmetry order, run ID, individual metrics and passing status.
Per-design normalization, per-ring evaluation and per-stack-grid caches also retain
completed work when later candidates fail or additional outputs arrive.
