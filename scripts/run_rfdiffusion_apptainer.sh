
#!/usr/bin/env bash
set -euo pipefail

NUM_DESIGNS="${RFD_NUM_DESIGNS:-1}"
OUTPUT_TAG="${RFD_OUTPUT_TAG:-design}"
INPUT_TAG="${RFD_INPUT_TAG:-input}"
MODEL="${RFD_MODEL:-active}"

# Validate settings
if ! [[ "$NUM_DESIGNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: RFD_NUM_DESIGNS must be a positive integer"
    exit 1
fi

if ! [[ "$OUTPUT_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "ERROR: Invalid RFD_OUTPUT_TAG"
    exit 1
fi


PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"

WORKROOT="/scratch/gpfs/JATKINSON/$USER"


CONTAINER="$WORKROOT/containers/rfdiffusion.sif"

if [[ ! -f "$CONTAINER" ]]; then
    CONTAINER="$PROJECT_ROOT/containers/rfdiffusion.sif"
fi


case "$MODEL" in
    active)
        CHECKPOINT_NAME="ActiveSite_ckpt.pt"
        ;;
    base)
        CHECKPOINT_NAME="Base_ckpt.pt"
        ;;
    *)
        echo "ERROR: RFD_MODEL must be active or base"
        exit 1
        ;;
esac

MODEL_DIR="$WORKROOT/models"

if [[ ! -f "$MODEL_DIR/$CHECKPOINT_NAME" ]]; then
    MODEL_DIR="$PROJECT_ROOT/models"
fi

CHECKPOINT="$MODEL_DIR/$CHECKPOINT_NAME"


INPUT_DIR="$PROJECT_ROOT/data/motifs/${INPUT_TAG}"

OUTPUT_DIR="$PROJECT_ROOT/outputs/rfdiffusion/1off_2fe2s/${OUTPUT_TAG}"

INPUT_PDB="$INPUT_DIR/rfdiffusion_input.pdb"

CONSTRAINTS="$INPUT_DIR/design_constraints.json"


for file in \
    "$CONTAINER" \
    "$CHECKPOINT" \
    "$INPUT_PDB" \
    "$CONSTRAINTS"
do
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

echo "========================================"
echo "Fe-S RFdiffusion Experiment"
echo "========================================"
echo "Project: $PROJECT_ROOT"
echo "Container: $CONTAINER"
echo "Model: $MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Input PDB: $INPUT_PDB"
echo "Contig: $CONTIG"
echo "Number of designs: $NUM_DESIGNS"
echo "Output tag: $OUTPUT_TAG"
echo "Output prefix: /output/$OUTPUT_TAG"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"


apptainer run --nv \
    --bind "$INPUT_DIR:/input:ro" \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$OUTPUT_DIR/schedules:/app/RFdiffusion/schedules" \
    --bind "$MODEL_DIR:/models:ro" \
    "$CONTAINER" \
    inference.input_pdb=/input/rfdiffusion_input.pdb \
    inference.output_prefix="/output/${OUTPUT_TAG}" \
    inference.model_directory_path=/models \
    inference.schedule_directory_path=/output/schedules \
    inference.ckpt_override_path="/models/${CHECKPOINT_NAME}" \
    inference.num_designs="${NUM_DESIGNS}" \
    inference.cautious=True \
    inference.write_trajectory=False \
    "$CONTIG"

echo "========================================"
echo "RFdiffusion finished."
echo "Model: $MODEL"
echo "Designs requested: $NUM_DESIGNS"
echo "Output tag: $OUTPUT_TAG"
echo "Results saved to: $OUTPUT_DIR"
echo "Done."
echo "========================================"
