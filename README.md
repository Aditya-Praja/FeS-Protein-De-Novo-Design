# De Novo Fe–S Protein Design

Computational design of symmetric protein oligomers that scaffold an experimentally observed [2Fe–2S] cluster, using generative structural models and geometry-based screening.

## Overview

This project explores the de novo design of protein scaffolds capable of accommodating iron–sulfur clusters. Fe–S clusters are important biological cofactors involved in electron transfer, catalysis, and other essential cellular processes.

The goal is to develop a computational pipeline that generates novel protein structures around a specified Fe–S coordination motif, arranges identical monomers into cyclic rings (C4 first; C3–C10 are configurable), stacks the rings into finite filament models using repeating rise/twist transformations, and screens the results geometrically. The motif comes from the 2Fe–2S ferredoxin 1OFF (*Synechocystis* sp. PCC 6803).

Sequence design (LigandMPNN) and experimental validation are later phases. Only the structure-input preparation for LigandMPNN is implemented so far.

This work serves as an initial foundation for investigating more unconventional metal–sulfur architectures, including proposed nickel–sulfur structures associated with electron transport in cable bacteria.

## Biological Background

Iron–sulfur clusters consist of iron atoms coordinated with inorganic sulfide ions and surrounding amino acid residues, commonly cysteine. Here the [2Fe–2S] cluster (residue `FES`) is extracted from PDB 1OFF, using a 2.8 Å cutoff to identify coordinating residues.

These clusters can support electron transfer through reversible changes in their oxidation states. Their function depends on both the cluster's chemical properties and the surrounding protein environment.

### Reference Protein Structure (1OFF)

![Reference Fe–S protein structure](docs/figures/reference.png)

### Extracted Fe–S Coordination Motif

![Fe2S2 Motif](docs/figures/Fe2S2.png)

![Extracted Fe–S coordination motif](docs/figures/motif_visualization.png)
## Computational Design Pipeline

The pipeline combines structural motif extraction, symmetric backbone generation, and geometric screening.

```text
Experimental Fe–S Protein Structure (1OFF)
                 |
                 v
       Fe–S Motif Extraction
                 |
                 v
   Motif Validation + Symmetric Input
   Preparation (C4 ring, rise/twist grid)
                 |
                 v
      RFdiffusion3 Backbone Generation
                 |
                 v
        Output Normalization
                 |
                 v
   Ring Screening -> Stack Screening
                 |
                 v
      Cofactor Graphs & Ranking
                 |
                 v
 LigandMPNN Structural Input Preparation
                 |
                 v
     Candidate Fe–S Oligomer Backbones
```

**1. Motif extraction:** Identify the Fe–S cluster and its coordinating residues from the experimental structure and validate the motif against the source.

**2. Symmetric input preparation:** Place identical copies of the motif into a cyclic ring (C4 pilot: 18 Å radius, 100-residue monomers) and define the rise/twist transformations used to stack rings (pilot grid: rises 20/25/30 Å, twists 0/15/30/45°, 3 layers).

**3. Backbone generation:** Run RFdiffusion3 (Foundry, pinned revision) with symmetry to generate monomers around the motif. Inference is run manually on a GPU (or CPU) Slurm node via `cluster/slurm/`.

**4. Geometric screening:** Normalize outputs, then measure motif preservation and clashes in rings and stacks, build cofactor graphs, and rank candidates.

**5. Next phase:** Prepare monomer, ring, and stack contexts for LigandMPNN sequence design. Sequence design, structure-prediction validation, and experimental work remain future steps.

Earlier exploratory work used RFdiffusion and ProteinMPNN on monomeric scaffolds (flank 2 vs 3 comparison). Those scripts and results are preserved in `scripts/`, `old_outputs/`, and `old_logs/`.

Geometry alone does not establish Fe–S binding, electron transfer, conductivity, affinity, thermodynamic stability, or self-assembly. Experimental characterization would be required to validate these properties.

## Designed Protein Structures

*Placeholder: Side-by-side comparison of the reference structure and a generated C4 ring scaffold.*

![Reference versus designed protein](docs/figures/design_comparison.png)

*Placeholder: Close-up comparison of the intended Fe–S coordination motif and its geometry in a selected designed scaffold.*

![Designed Fe–S binding site](docs/figures/designed_binding_site.png)

## Project Organization

```text
configs/       architecture YAML (pilot, production) and Della cluster settings
data/          raw PDB (1off) and extracted motif files
src/fes_wire/  preparation, geometry, ranking, RFD3 adapters and CLI
scripts/       motif extraction, cluster runner/validation, earlier RFdiffusion/ProteinMPNN scripts
runs/          per-run stage results (motif validation, RFD3 inputs, ...)
cluster/       Slurm job scripts (RFD3 GPU/CPU, earlier RFdiffusion/ProteinMPNN)
external/      RFdiffusion and ProteinMPNN submodules
old_outputs/   earlier flank 2/3 RFdiffusion evaluation results
old_logs/      earlier Slurm logs
docs/          figures
tests/         synthetic geometry and interface regression tests
```

Quick start (from the repository root, Python 3.9–3.12 for local analysis; RFD3 inference needs Python 3.12):

```bash
python3 -m venv .venv-wire && source .venv-wire/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pip install --no-deps -e .
fes-wire pre-rfd --config configs/architectures/c4_pilot.yaml
python -m pytest -q
```

RFD3 inference on Della (do not pass `--partition`; the cluster routes the job):

```bash
export FES_RUN_ID=c4_pilot RFD3_ENV=/path/to/env RFD3_CHECKPOINT=/path/to/checkpoint
sbatch --array=0 --export=ALL cluster/slurm/rfd3_cpu.sbatch   # CPU
# GPU: source configs/cluster/della.env, then sbatch with --constraint/--mem/--time/--cpus-per-task and cluster/slurm/rfd3.sbatch
```

## Computational Tools

| Tool | Role |
|---|---|
| BioPython | Structure parsing, motif extraction, SASA |
| NumPy / SciPy / NetworkX | Geometry, symmetry transforms, cofactor graphs |
| RFdiffusion3 (Foundry) | Symmetric de novo backbone generation |
| LigandMPNN | Planned sequence design (inputs prepared) |
| RFdiffusion / ProteinMPNN | Earlier exploratory monomer workflow |
| PyTorch | Deep learning inference |
| PyMOL | Molecular visualization |
| Slurm (Della) | Cluster execution |

## Personal Notes



## References:
Approximate Structural Values for 2Fe-2S: [The competition between chemistry and biology in assembling iron–sulfur derivatives](https://www.sciencedirect.com/science/article/abs/pii/S0010854514002082)