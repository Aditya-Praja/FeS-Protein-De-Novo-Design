
#!/usr/bin/env bash
set -euo pipefail
NUM_DESIGNS="${NUM_DESIGNS:-1}"

# Project directories
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
WORKROOT="/scratch/gpfs/JATKINSON/$USER"

# RFdiffusion container
CONTAINER="$WORKROOT/containers/rfdiffusion.sif"

# Support a container stored inside the repository
if [[ ! -f "$CONTAINER" ]]; then
    CONTAINER="$PROJECT_ROOT/containers/rfdiffusion.sif"
fi

# Model checkpoints
MODEL_DIR="$WORKROOT/models"

if [[ ! -f "$MODEL_DIR/ActiveSite_ckpt.pt" ]]; then
    MODEL_DIR="$PROJECT_ROOT/models"
fi

CHECKPOINT="$MODEL_DIR/ActiveSite_ckpt.pt"

# Motif input and output
INPUT_DIR="$PROJECT_ROOT/data/motifs/1off_2fe2s"
OUTPUT_DIR="$PROJECT_ROOT/outputs/rfdiffusion/1off_2fe2s"

INPUT_PDB="$INPUT_DIR/rfdiffusion_input.pdb"
CONSTRAINTS="$INPUT_DIR/design_constraints.json"
OUTPUT_TAG="${OUTPUT_TAG:-design}"

for file in "$CONTAINER" "$CHECKPOINT" \
            "$INPUT_PDB" "$CONSTRAINTS"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Missing required file: $file"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/schedules"

CONTIG=$(python3 - "$CONSTRAINTS" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    metadata = json.load(f)

print(metadata["rfdiffusion_contig"])
PY
)

echo "Starting RFdiffusion"
echo "Input PDB: $INPUT_PDB"
echo "Checkpoint: $CHECKPOINT"
echo "Contig: $CONTIG"
echo "Output: $OUTPUT_DIR"
echo "Number of designs: $NUM_DESIGNS"

echo "Number of designs: $NUM_DESIGNS"
echo "Output tag: $OUTPUT_TAG"

apptainer run --nv \
    --bind "$INPUT_DIR:/input:ro" \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$OUTPUT_DIR/schedules:/app/RFdiffusion/schedules" \
    --bind "$MODEL_DIR:/models:ro" \
    "$CONTAINER" \
    inference.input_pdb=/input/rfdiffusion_input.pdb \
    inference.output_prefix=/output/"$OUTPUT_TAG" \
    inference.model_directory_path=/models \
    inference.schedule_directory_path=/output/schedules \
    inference.ckpt_override_path=/models/ActiveSite_ckpt.pt \
    inference.num_designs="$NUM_DESIGNS" \
    inference.write_trajectory=False \
    "$CONTIG"

echo "RFdiffusion finished."
echo "Results saved to: $OUTPUT_DIR"
