#!/usr/bin/env python
"""
Extract Gene Column Names from trt_sh_qc Dataset

Extracts the list of transcriptomic feature columns (Entrez gene IDs) from the
filtered shRNA knockdown dataset and saves them to a text file. This list is
used by downstream scripts (04_2) to identify gene expression features without
needing to filter out metadata columns.

Input:
    ${CONTEXTPERT_DATA_DIR}/trt_sh_qc.csv

Output:
    ${CONTEXTPERT_DATA_DIR}/trt_sh_qc_gene_cols.txt
"""

import os
import pandas as pd

DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')

print("=" * 80)
print("EXTRACTING GENE COLUMNS FROM trt_sh_qc")
print("=" * 80)

# Load trt_sh_qc data (just header to get column names)
input_path = os.path.join(DATA_DIR, 'trt_sh_qc.csv')
print(f"\nLoading trt_sh_qc header from: {input_path}")

# Read just the first row to get column names efficiently
df_header = pd.read_csv(input_path, nrows=0)

print(f"  Total columns: {len(df_header.columns)}")

# Define metadata columns (non-gene columns)
metadata_cols = [
    'inst_id',
    'cell_id',
    'pert_id',
    'pert_type',
    'pert_dose',
    'pert_dose_unit',
    'pert_time',
    'sig_id',
    'distil_cc_q75',
    'pct_self_rank_q25'
]

print(f"\nMetadata columns: {len(metadata_cols)}")
print(f"  {metadata_cols}")

# Extract gene columns (all non-metadata columns)
gene_cols = [col for col in df_header.columns if col not in metadata_cols]

print(f"\nGene columns (Entrez IDs): {len(gene_cols)}")
print(f"  First 10: {gene_cols[:10]}")
print(f"  Last 10: {gene_cols[-10:]}")

# Verify gene columns look like numeric Entrez IDs
numeric_pattern_count = sum(1 for col in gene_cols if col.isdigit())
print(f"\n  Numeric columns (Entrez IDs): {numeric_pattern_count}/{len(gene_cols)} ({numeric_pattern_count/len(gene_cols)*100:.1f}%)")

# Save gene columns to text file
output_path = os.path.join(DATA_DIR, 'trt_sh_qc_gene_cols.txt')
print(f"\nSaving gene column names to: {output_path}")

with open(output_path, 'w') as f:
    for gene_col in gene_cols:
        f.write(f"{gene_col}\n")

print(f"  Saved {len(gene_cols)} gene column names")

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
print(f"\nOutput: {output_path}")
print(f"  Total gene columns: {len(gene_cols)}")
print("\n Done")
