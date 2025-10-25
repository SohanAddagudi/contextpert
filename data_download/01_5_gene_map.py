#!/usr/bin/env python
"""
Create Gene ID Mapping: Entrez to Ensembl

Uses mygene.info API to map Entrez gene IDs to Ensembl gene IDs and symbols.
Extracts Entrez IDs from the filtered LINCS dataset columns and creates a
comprehensive mapping for downstream use.

Input:
    ${CONTEXTPERT_DATA_DIR}/trt_sh_qc.csv (or trt_cp_smiles_qc.csv)

Output:
    ${CONTEXTPERT_DATA_DIR}/entrez_to_ensembl_map.csv
    Columns: entrez_id, ensembl_id, symbol
"""

import os
import pandas as pd
import mygene

DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')

print("=" * 80)
print("CREATING ENTREZ TO ENSEMBL GENE MAPPING")
print("=" * 80)

# Load trt_sh_qc data to get Entrez gene IDs from columns
input_path = os.path.join(DATA_DIR, 'trt_sh_qc.csv')
print(f"\nLoading gene columns from: {input_path}")

# Read just the header to get column names
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

# Extract gene columns (Entrez IDs)
gene_cols = [col for col in df_header.columns if col not in metadata_cols]

print(f"\nExtracted gene columns:")
print(f"  Metadata columns: {len(metadata_cols)}")
print(f"  Gene columns (Entrez IDs): {len(gene_cols)}")
print(f"  First 10 Entrez IDs: {gene_cols[:10]}")
print(f"  Last 10 Entrez IDs: {gene_cols[-10:]}")

# Query mygene.info API for mapping
print("\n" + "=" * 80)
print("QUERYING MYGENE.INFO API")
print("=" * 80)

print(f"\nQuerying mygene.info for {len(gene_cols)} Entrez IDs...")
print("  Fields: ensembl.gene, symbol")
print("  Species: human")

mg = mygene.MyGeneInfo()
gene_info = mg.querymany(
    gene_cols,
    scopes="entrezgene",
    fields="ensembl.gene,symbol",
    species="human",
    as_dataframe=False,
    returnall=True
)

print(f"\nAPI response received")
print(f"  Total results: {len(gene_info['out'])}")

# Process results into mapping dataframe
print("\n" + "=" * 80)
print("PROCESSING MAPPING RESULTS")
print("=" * 80)

mapping_data = []
unmapped_count = 0
multiple_ensembl_count = 0

for item in gene_info['out']:
    entrez_id = str(item.get('query', ''))

    # Get symbol
    symbol = item.get('symbol', None)

    # Get Ensembl ID (handle both single ID and list of IDs)
    ensembl_data = item.get('ensembl', None)
    ensembl_id = None

    if ensembl_data:
        if isinstance(ensembl_data, list):
            # Multiple Ensembl IDs - take the first one
            ensembl_id = ensembl_data[0].get('gene', None) if len(ensembl_data) > 0 else None
            if len(ensembl_data) > 1:
                multiple_ensembl_count += 1
        elif isinstance(ensembl_data, dict):
            # Single Ensembl ID
            ensembl_id = ensembl_data.get('gene', None)

    if ensembl_id and symbol:
        mapping_data.append({
            'entrez_id': entrez_id,
            'ensembl_id': ensembl_id,
            'symbol': symbol
        })
    else:
        unmapped_count += 1

print(f"\nMapping statistics:")
print(f"  Total Entrez IDs queried: {len(gene_cols)}")
print(f"  Successfully mapped: {len(mapping_data)}")
print(f"  Unmapped: {unmapped_count}")
print(f"  Mapping coverage: {len(mapping_data)/len(gene_cols)*100:.1f}%")

if multiple_ensembl_count > 0:
    print(f"\n  Note: {multiple_ensembl_count} Entrez IDs had multiple Ensembl IDs (used first)")

# Create dataframe
mapping_df = pd.DataFrame(mapping_data)

print(f"\nMapping dataframe:")
print(f"  Shape: {mapping_df.shape}")
print(f"  Columns: {list(mapping_df.columns)}")
print(f"\nFirst 10 mappings:")
print(mapping_df.head(10).to_string(index=False))

# Save mapping
output_path = os.path.join(DATA_DIR, 'entrez_to_ensembl_map.csv')
print(f"\nSaving mapping to: {output_path}")
mapping_df.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("MAPPING CREATION COMPLETE")
print("=" * 80)
print(f"\nOutput: {output_path}")
print(f"  Total mappings: {len(mapping_df):,}")
print(f"  Unique Entrez IDs: {mapping_df['entrez_id'].nunique():,}")
print(f"  Unique Ensembl IDs: {mapping_df['ensembl_id'].nunique():,}")
print(f"  Unique symbols: {mapping_df['symbol'].nunique():,}")
print("\n✓ Done")
