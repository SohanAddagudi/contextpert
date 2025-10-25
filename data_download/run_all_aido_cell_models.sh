#!/bin/bash

# Script to generate embeddings for all AIDO.Cell backbone models
# Usage: bash run_all_aido_cell_models.sh [chunk_size]
# Default chunk size: 10000 rows

set -e  # Exit on error

# Get chunk size from command line argument or use default
CHUNK_SIZE=${1:-50000}

echo "Starting AIDO.Cell embedding generation for all backbone models..."
echo "Chunk size: $CHUNK_SIZE rows"
echo "=============================================================="

# Array of backbone models
BACKBONES=("aido_cell_100m" "aido_cell_10m" "aido_cell_3m")

# Run for each backbone
for BACKBONE in "${BACKBONES[@]}"; do
    echo ""
    echo "Processing backbone: $BACKBONE"
    echo "--------------------------------------------------------------"
    python 13_aido_cell_lincs_embedding.py --backbone "$BACKBONE" --chunk-size "$CHUNK_SIZE"
    echo "Completed: $BACKBONE"
    echo ""
done

echo "=============================================================="
echo "All AIDO.Cell models processed successfully!"
