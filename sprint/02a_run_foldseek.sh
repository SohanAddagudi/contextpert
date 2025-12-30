#!/bin/bash
# =============================================================================
# Run FoldSeek to Generate Structure-Aware Sequences
# =============================================================================
#
# This script converts AlphaFold structures to SaProt structure-aware sequences
# using FoldSeek (via SPRINT's structure_to_saprot.py).
#
# Prerequisites:
#   1. Run 01_prepare_inputs.py first (without --no-structure flag)
#   2. Install FoldSeek: https://github.com/steineggerlab/foldseek
#   3. Have SPRINT repo cloned
#
# =============================================================================

set -e

# Check environment variables
if [ -z "$CONTEXTPERT_DATA_DIR" ]; then
    echo "ERROR: CONTEXTPERT_DATA_DIR not set"
    exit 1
fi

if [ -z "$SPRINT_DIR" ]; then
    echo "ERROR: SPRINT_DIR not set (path to panspecies-dti repo)"
    exit 1
fi

# Convert to absolute paths (critical - we'll cd to SPRINT_DIR later)
SPRINT_DATA_DIR="$(cd "$CONTEXTPERT_DATA_DIR" && pwd)/sprint"
STRUCTURES_DIR="${SPRINT_DATA_DIR}/alphafold_structures"
STRUCTURE_MAPPING="${SPRINT_DATA_DIR}/structure_mapping.csv"
OUTPUT_CSV="${SPRINT_DATA_DIR}/targets_foldseek.csv"
FINAL_TARGETS="${SPRINT_DATA_DIR}/targets.csv"

# Check inputs
if [ ! -d "$STRUCTURES_DIR" ]; then
    echo "ERROR: Structures directory not found: $STRUCTURES_DIR"
    echo "Run 01_prepare_inputs.py first (without --no-structure flag)"
    exit 1
fi

if [ ! -f "$STRUCTURE_MAPPING" ]; then
    echo "ERROR: Structure mapping not found: $STRUCTURE_MAPPING"
    echo "Run 01_prepare_inputs.py first (without --no-structure flag)"
    exit 1
fi

# Check foldseek is installed
if ! command -v foldseek &> /dev/null; then
    echo "ERROR: foldseek not found in PATH"
    echo "Install from: https://github.com/steineggerlab/foldseek"
    exit 1
fi

# Count structures first (before cd)
TOTAL=$(ls "$STRUCTURES_DIR"/*.cif 2>/dev/null | wc -l)

echo "========================================"
echo "FOLDSEEK STRUCTURE-AWARE SEQUENCE GENERATION"
echo "========================================"
echo "Structures directory: $STRUCTURES_DIR"
echo "Structure files found: $TOTAL"
echo "Output: $OUTPUT_CSV"
echo ""

# Change to SPRINT directory (needed for structure_to_saprot.py imports)
cd "$SPRINT_DIR"

# Initialize output file (will be appended to)
rm -f "$OUTPUT_CSV"

echo "Processing $TOTAL structure files..."
echo ""

COUNT=0
FAILED=0

for CIF_FILE in "$STRUCTURES_DIR"/*.cif; do
    if [ ! -f "$CIF_FILE" ]; then
        continue
    fi
    
    COUNT=$((COUNT + 1))
    BASENAME=$(basename "$CIF_FILE")
    
    echo "[$COUNT/$TOTAL] Processing: $BASENAME"
    
    # Run structure_to_saprot.py
    # Using chain A
    if python utils/structure_to_saprot.py \
        -I "$CIF_FILE" \
        --chain A \
        -O "$OUTPUT_CSV" 2>/dev/null; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "FOLDSEEK PROCESSING COMPLETE"
echo "========================================"
echo "Processed: $COUNT"
echo "Failed: $FAILED"
echo ""

if [ ! -f "$OUTPUT_CSV" ]; then
    echo "ERROR: No output generated. Check FoldSeek installation."
    exit 1
fi

# Count sequences generated
SEQ_COUNT=$(wc -l < "$OUTPUT_CSV")
SEQ_COUNT=$((SEQ_COUNT - 1))  # Subtract header
echo "Generated $SEQ_COUNT structure-aware sequences"
echo "Output: $OUTPUT_CSV"

# Now we need to create the final targets.csv by merging with our mapping
echo ""
echo "========================================"
echo "CREATING FINAL TARGETS.CSV"
echo "========================================"

python << EOF
import pandas as pd
import os

# Load structure mapping (has target_id, uniprot_id, gene_symbol, structure_path)
mapping_df = pd.read_csv("$STRUCTURE_MAPPING")

# Load FoldSeek output (has Target Sequence column, one per structure processed)
foldseek_df = pd.read_csv("$OUTPUT_CSV")

# The FoldSeek output is in order of processing
# We need to match by structure file name

# Extract UniProt ID from structure path
mapping_df['structure_basename'] = mapping_df['structure_path'].apply(os.path.basename)

# Create mapping from basename to row index
# FoldSeek appends in order of processing, which matches our loop order
# So we can align by sorted structure paths

# Get list of structure basenames in the order they were processed
import glob
structure_files = sorted(glob.glob("$STRUCTURES_DIR/*.cif"))
structure_basenames = [os.path.basename(f) for f in structure_files]

# Check if counts match
if len(foldseek_df) != len(structure_basenames):
    print(f"Warning: FoldSeek output ({len(foldseek_df)}) != structure files ({len(structure_basenames)})")
    print("Some structures may have failed. Attempting best-effort matching...")

# For now, assume 1:1 correspondence if counts match
if len(foldseek_df) == len(structure_basenames):
    foldseek_df['structure_basename'] = structure_basenames
    
    # Merge with mapping
    merged = mapping_df.merge(
        foldseek_df[['structure_basename', 'Target Sequence']], 
        on='structure_basename',
        how='inner'
    )
    
    # Create final targets.csv
    final_df = merged[['target_id', 'uniprot_id', 'gene_symbol', 'Target Sequence']].copy()
    final_df.to_csv("$FINAL_TARGETS", index=False)
    
    print(f"Created: $FINAL_TARGETS")
    print(f"  Targets: {len(final_df)}")
    print(f"  Columns: {list(final_df.columns)}")
else:
    print("ERROR: Could not align FoldSeek output with structure files")
    print("Please check for errors in FoldSeek processing")
    exit(1)
EOF

