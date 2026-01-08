#!/bin/bash
# =============================================================================
# Run SPRINT Embedding
# =============================================================================
#
# This script runs SPRINT to generate drug and target embeddings.
#
# Prerequisites:
#   1. Install SPRINT:
#      conda create -n sprint python=3.10 -y
#      conda activate sprint
#      pip install git+https://github.com/abhinadduri/panspecies-dti.git
#
#   2. Download checkpoint:
#      See https://github.com/abhinadduri/panspecies-dti/blob/main/checkpoints/README.md
#      Download sprint.ckpt and place in checkpoints/
#
#   3. Run 01_prepare_inputs.py first to generate drugs.csv and targets.csv
#
# =============================================================================

set -e  # Exit on error

export CONTEXTPERT_DATA_DIR = "Path to data directory"
export SPRINT_DIR = "Path to panspecies-dti repo"

# Check environment variables
if [ -z "$CONTEXTPERT_DATA_DIR" ]; then
    echo "ERROR: CONTEXTPERT_DATA_DIR not set"
    exit 1
fi

if [ -z "$SPRINT_DIR" ]; then
    echo "ERROR: SPRINT_DIR not set (path to panspecies-dti repo)"
    exit 1
fi

SPRINT_DATA_DIR="${CONTEXTPERT_DATA_DIR}/sprint"
CHECKPOINT="${SPRINT_DIR}/checkpoints/sprint.ckpt"

# Check inputs exist
if [ ! -f "${SPRINT_DATA_DIR}/drugs.csv" ]; then
    echo "ERROR: ${SPRINT_DATA_DIR}/drugs.csv not found"
    echo "Run 01_prepare_inputs.py first"
    exit 1
fi

if [ ! -f "${SPRINT_DATA_DIR}/targets.csv" ]; then
    echo "ERROR: ${SPRINT_DATA_DIR}/targets.csv not found"
    echo "Run 01_prepare_inputs.py first"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Download from: https://github.com/abhinadduri/panspecies-dti/blob/main/checkpoints/README.md"
    exit 1
fi

echo "========================================"
echo "SPRINT EMBEDDING"
echo "========================================"
echo "Data directory: $SPRINT_DATA_DIR"
echo "SPRINT directory: $SPRINT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo ""

# Change to SPRINT directory
cd "$SPRINT_DIR"

# Embed drugs
echo "----------------------------------------"
echo "Step 1: Embedding drugs..."
echo "----------------------------------------"
ultrafast-embed \
    --data-file "${SPRINT_DATA_DIR}/drugs.csv" \
    --checkpoint "$CHECKPOINT" \
    --moltype drug \
    --output-path "${SPRINT_DATA_DIR}/drug_embeddings.npy"

echo "✓ Drug embeddings saved to: ${SPRINT_DATA_DIR}/drug_embeddings.npy"

# Embed targets
echo ""
echo "----------------------------------------"
echo "Step 2: Embedding targets..."
echo "----------------------------------------"
ultrafast-embed \
    --data-file "${SPRINT_DATA_DIR}/targets.csv" \
    --checkpoint "$CHECKPOINT" \
    --moltype target \
    --output-path "${SPRINT_DATA_DIR}/target_embeddings.npy"

echo "✓ Target embeddings saved to: ${SPRINT_DATA_DIR}/target_embeddings.npy"

echo ""
echo "========================================"
echo "EMBEDDING COMPLETE"
echo "========================================"
echo ""
echo "Outputs:"
echo "  ${SPRINT_DATA_DIR}/drug_embeddings.npy"
echo "  ${SPRINT_DATA_DIR}/target_embeddings.npy"

