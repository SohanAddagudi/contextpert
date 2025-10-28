#!/usr/bin/env python
"""
Add Target Annotations to Cell Embedding Files

Adds gene_symbol and ensembl_id columns to the embedded trt_sh files,
creating the files as they would have looked if trt_sh_genes_qc.csv
had been used as input to the embedding script.

Input:
    ${CONTEXTPERT_DATA_DIR}/trt_sh_genes_qc.csv
    ${CONTEXTPERT_DATA_DIR}/trt_sh_qc_aido_cell_*_embeddings.csv

Output:
    ${CONTEXTPERT_DATA_DIR}/trt_sh_genes_qc_aido_cell_*_embeddings.csv
"""

import os
import pandas as pd

DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')

print("=" * 80)
print("ADDING TARGET ANNOTATIONS TO EMBEDDING FILES")
print("=" * 80)

# Load target annotations
print("\nLoading target annotations...")
trt_sh_genes_path = os.path.join(DATA_DIR, 'trt_sh_genes_qc.csv')
trt_sh_genes_df = pd.read_csv(trt_sh_genes_path, low_memory=False)

# Extract unique pert_id -> gene_symbol, ensembl_id mapping
target_mapping = trt_sh_genes_df[['pert_id', 'gene_symbol', 'ensembl_id']].drop_duplicates()
print(f"  Loaded mappings for {len(target_mapping)} unique pert_ids")
print(f"  Unique gene symbols: {target_mapping['gene_symbol'].nunique()}")
print(f"  Unique ensembl_ids: {target_mapping['ensembl_id'].nunique()}")

# Process each embedding file
embedding_models = ['3m', '10m', '100m']

for model in embedding_models:
    print("\n" + "=" * 80)
    print(f"PROCESSING: aido_cell_{model}")
    print("=" * 80)

    # Load embedding file
    input_path = os.path.join(DATA_DIR, f'trt_sh_qc_aido_cell_{model}_embeddings.csv')
    print(f"\nLoading: {input_path}")
    embed_df = pd.read_csv(input_path)

    print(f"  Rows: {len(embed_df):,}")
    print(f"  Columns: {len(embed_df.columns)}")
    print(f"  Unique pert_ids: {embed_df['pert_id'].nunique():,}")

    # Merge with target annotations
    print("\nMerging with target annotations...")
    merged_df = embed_df.merge(target_mapping, on='pert_id', how='left')

    print(f"  Merged rows: {len(merged_df):,}")
    print(f"  Rows with gene_symbol: {merged_df['gene_symbol'].notna().sum():,}")
    print(f"  Rows with ensembl_id: {merged_df['ensembl_id'].notna().sum():,}")

    # Reorder columns to put gene_symbol and ensembl_id after metadata, before embeddings
    metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                     'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                     'pct_self_rank_q25']
    target_cols = ['gene_symbol', 'ensembl_id']
    embedding_cols = [col for col in merged_df.columns if col.startswith('emb_')]

    # Final column order
    final_cols = metadata_cols + target_cols + embedding_cols
    merged_df = merged_df[final_cols]

    print(f"\nFinal column order:")
    print(f"  Metadata: {len(metadata_cols)} columns")
    print(f"  Target annotations: {len(target_cols)} columns")
    print(f"  Embeddings: {len(embedding_cols)} columns")

    # Save with new filename
    output_path = os.path.join(DATA_DIR, f'trt_sh_genes_qc_aido_cell_{model}_embeddings.csv')
    print(f"\nSaving to: {output_path}")
    merged_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(merged_df):,} rows")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)

print("\nCreated files:")
for model in embedding_models:
    output_path = os.path.join(DATA_DIR, f'trt_sh_genes_qc_aido_cell_{model}_embeddings.csv')
    print(f"  {output_path}")

print("\n✓ Done")
