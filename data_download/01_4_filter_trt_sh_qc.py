#!/usr/bin/env python
"""
Filter trt_sh with Quality Control and Map Gene IDs

Applies the standard LINCS quality control filter to shRNA knockdown data and
maps Entrez gene IDs to Ensembl gene IDs (ENSG) for compatibility with
OpenTargets.

Input:
    ${CONTEXTPERT_DATA_DIR}/trt_sh.csv
    ${CONTEXTPERT_DATA_DIR}/hgnc/hgnc_complete_set.txt

Output:
    ${CONTEXTPERT_DATA_DIR}/trt_sh_qc.csv (filtered data with gene columns)
    ${CONTEXTPERT_DATA_DIR}/entrez_to_ensembl_map.csv (ID mapping reference)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')

print("=" * 80)
print("FILTERING trt_sh WITH QUALITY CONTROL AND GENE MAPPING")
print("=" * 80)

# Load HGNC data for gene ID mapping
print("\n" + "=" * 80)
print("LOADING HGNC GENE MAPPING DATA")
print("=" * 80)

hgnc_path = os.path.join(DATA_DIR, 'hgnc/hgnc_complete_set.txt')
print(f"\nLoading HGNC data from: {hgnc_path}")
hgnc_df = pd.read_csv(hgnc_path, sep='\t', low_memory=False)

print(f"Loaded HGNC data:")
print(f"  Total entries: {len(hgnc_df):,}")
print(f"  Columns: {hgnc_df.columns.tolist()[:10]}...")

# Extract Entrez ID to Ensembl ID mapping
print("\nExtracting Entrez ID -> Ensembl ID mapping...")

# HGNC has entrez_id and ensembl_gene_id columns
entrez_col = 'entrez_id'
ensembl_col = 'ensembl_gene_id'

if entrez_col not in hgnc_df.columns or ensembl_col not in hgnc_df.columns:
    print(f"Available columns: {hgnc_df.columns.tolist()}")
    raise ValueError(f"Required columns '{entrez_col}' and '{ensembl_col}' not found in HGNC data")

# Filter to rows with both IDs present
mapping_df = hgnc_df[[entrez_col, ensembl_col, 'symbol']].copy()
mapping_df = mapping_df[mapping_df[entrez_col].notna() & mapping_df[ensembl_col].notna()].copy()

# Convert Entrez IDs to integers first, then to strings (they're stored as column names in trt_sh)
# HGNC stores them as floats like "1.0" but trt_sh columns are like "55847"
mapping_df[entrez_col] = mapping_df[entrez_col].astype(float).astype(int).astype(str)

# Remove any with empty strings
mapping_df = mapping_df[mapping_df[entrez_col] != '']
mapping_df = mapping_df[mapping_df[ensembl_col] != '']

print(f"Created mapping:")
print(f"  Genes with both Entrez and Ensembl IDs: {len(mapping_df):,}")

# Create mapping dictionary: entrez_id -> ensembl_id
entrez_to_ensembl = dict(zip(mapping_df[entrez_col], mapping_df[ensembl_col]))
entrez_to_symbol = dict(zip(mapping_df[entrez_col], mapping_df['symbol']))

print(f"  Mapping dictionary size: {len(entrez_to_ensembl):,}")
print(f"\nExample mappings:")
for i, (entrez, ensembl) in enumerate(list(entrez_to_ensembl.items())[:5]):
    symbol = entrez_to_symbol.get(entrez, 'N/A')
    print(f"  Entrez {entrez} ({symbol}) -> {ensembl}")

# Save mapping for reference
mapping_output_path = os.path.join(DATA_DIR, 'entrez_to_ensembl_map.csv')
print(f"\nSaving mapping to: {mapping_output_path}")
mapping_df.to_csv(mapping_output_path, index=False)

# Load trt_sh data
print("\n" + "=" * 80)
print("LOADING shRNA KNOCKDOWN DATA")
print("=" * 80)

input_path = os.path.join(DATA_DIR, 'trt_sh.csv')
print(f"\nLoading shRNA perturbation data from: {input_path}")
df = pd.read_csv(input_path, low_memory=False)

print(f"Loaded trt_sh:")
print(f"  Total samples: {len(df):,}")
print(f"  Unique perturbation IDs: {df['pert_id'].nunique():,}")
print(f"  Shape: {df.shape}")

# Identify metadata and gene columns
metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                 'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                 'pct_self_rank_q25']

gene_cols = [col for col in df.columns if col not in metadata_cols]

print(f"\nColumn breakdown:")
print(f"  Metadata columns: {len(metadata_cols)}")
print(f"  Gene columns (Entrez IDs): {len(gene_cols)}")
print(f"  First 10 gene columns: {gene_cols[:10]}")

# Apply quality control filter
print("\n" + "=" * 80)
print("APPLYING QUALITY CONTROL FILTER")
print("=" * 80)

print("\nFiltering by quality control metrics...")
print("Removing samples where:")
print("  - distil_cc_q75 < 0.2 OR == -666 OR is NaN")
print("  - pct_self_rank_q25 > 5 OR == -666 OR is NaN")

condition = (
    (df['distil_cc_q75'] < 0.2) |
    (df['distil_cc_q75'] == -666) |
    (df['distil_cc_q75'].isna()) |
    (df['pct_self_rank_q25'] > 5) |
    (df['pct_self_rank_q25'] == -666) |
    (df['pct_self_rank_q25'].isna())
)

n_removed = condition.sum()
df_filtered = df[~condition].copy()

print(f"\nQuality control results:")
print(f"  Removed: {n_removed:,} low-quality samples")
print(f"  Remaining: {len(df_filtered):,} samples")
print(f"  Removal rate: {n_removed/len(df)*100:.1f}%")

# Map gene columns from Entrez to Ensembl
print("\n" + "=" * 80)
print("MAPPING GENE COLUMNS: ENTREZ -> ENSEMBL")
print("=" * 80)

print(f"\nMapping {len(gene_cols)} gene columns from Entrez IDs to Ensembl IDs...")

# Check which gene columns can be mapped
mappable_genes = [col for col in gene_cols if col in entrez_to_ensembl]
unmappable_genes = [col for col in gene_cols if col not in entrez_to_ensembl]

print(f"  Mappable genes: {len(mappable_genes)}")
print(f"  Unmappable genes: {len(unmappable_genes)}")
print(f"  Mapping coverage: {len(mappable_genes)/len(gene_cols)*100:.1f}%")

if len(unmappable_genes) > 0:
    print(f"\n  First 10 unmappable Entrez IDs: {unmappable_genes[:10]}")

# Create new dataframe with metadata and mappable gene columns only
print(f"\nCreating filtered dataframe with only mappable genes...")

# Keep metadata columns
output_df = df_filtered[metadata_cols].copy()

# Add gene columns with Ensembl IDs as column names
print(f"  Renaming {len(mappable_genes)} gene columns to Ensembl IDs...")
for entrez_id in mappable_genes:
    ensembl_id = entrez_to_ensembl[entrez_id]
    output_df[ensembl_id] = df_filtered[entrez_id]

print(f"  New gene columns (first 10): {list(output_df.columns[len(metadata_cols):len(metadata_cols)+10])}")

# Final statistics
print("\n" + "=" * 80)
print("FINAL STATISTICS")
print("=" * 80)

print(f"\nOriginal data:")
print(f"  Samples: {len(df):,}")
print(f"  Unique perturbation IDs: {df['pert_id'].nunique():,}")
print(f"  Gene columns (Entrez): {len(gene_cols)}")

print(f"\nFiltered data:")
print(f"  Samples: {len(output_df):,}")
print(f"  Unique perturbation IDs: {output_df['pert_id'].nunique():,}")
print(f"  Gene columns (Ensembl): {len(mappable_genes)}")
print(f"  Removed unmappable genes: {len(unmappable_genes)}")

print(f"\nOverall changes:")
print(f"  Sample removal: {len(df) - len(output_df):,} ({(len(df)-len(output_df))/len(df)*100:.1f}%)")
print(f"  Gene retention: {len(mappable_genes)}/{len(gene_cols)} ({len(mappable_genes)/len(gene_cols)*100:.1f}%)")

# Save filtered data
output_path = os.path.join(DATA_DIR, 'trt_sh_qc.csv')
print(f"\nSaving filtered data to: {output_path}")
output_df.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("FILTERING COMPLETE")
print("=" * 80)
print(f"\nOutputs:")
print(f"  1. {output_path}")
print(f"     - Filtered samples: {len(output_df):,}")
print(f"     - Unique perturbation IDs: {output_df['pert_id'].nunique():,}")
print(f"     - Gene columns (ENSG IDs): {len(mappable_genes)}")
print(f"\n  2. {mapping_output_path}")
print(f"     - Entrez to Ensembl mappings: {len(mapping_df):,}")
print("\n✓ Done")
