#!/usr/bin/env python
"""
Annotate trt_sh with genetic targets

Input:
    ${CONTEXTPERT_DATA_DIR}/trt_sh_qc.csv
    ${CONTEXTPERT_DATA_DIR}/pert_target.csv

Output:
    ${CONTEXTPERT_DATA_DIR}/trt_sh_genes_qc.csv
"""

import os
import sys
import pandas as pd
from pathlib import Path
import mygene

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from contextpert.utils import canonicalize_smiles

DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')

# Load LINCS shRNA data (targets) via pert_target mapping
print("\n" + "=" * 80)
print("LOADING LINCS TARGET DATA")
print("=" * 80)

# Load pert_target data to get true trt_sh targets
pert_target_path = os.path.join(DATA_DIR, 'pert_target.csv')
print(f"\nLoading pert_target data from: {pert_target_path}")
pert_target_df = pd.read_csv(pert_target_path)

# Filter for trt_sh perturbations
trt_sh_targets = pert_target_df[pert_target_df['pert_type'] == 'trt_sh'].copy()
print(f"Loaded pert_target data:")
print(f"  Total pert_target entries: {len(pert_target_df):,}")
print(f"  trt_sh perturbations: {len(trt_sh_targets):,}")
print(f"  Unique trt_sh pert_ids: {trt_sh_targets['pert_id'].nunique():,}")
print(f"  Unique gene symbols: {trt_sh_targets['pert_iname'].nunique():,}")

# Extract unique gene symbols from trt_sh targets
gene_symbols = trt_sh_targets['pert_iname'].dropna().unique().tolist()
print(f"\nUnique gene symbols to map: {len(gene_symbols):,}")
print(f"  Example symbols: {gene_symbols[:10]}")

# Map gene symbols to Ensembl IDs using mygene
print(f"\nMapping gene symbols to Ensembl IDs using mygene...")
mg = mygene.MyGeneInfo()
gene_info = mg.querymany(
    gene_symbols,
    scopes='symbol',
    fields='ensembl.gene',
    species='human',
    as_dataframe=False,
    returnall=True
)

# Process mygene results
symbol_to_ensembl = {}
unmapped_symbols = []

for result in gene_info['out']:
    if 'ensembl' in result and 'notfound' not in result:
        symbol = result['query']
        ensembl_data = result['ensembl']

        # Handle both single and multiple Ensembl IDs
        if isinstance(ensembl_data, list):
            # Take the first Ensembl ID if multiple
            ensembl_id = ensembl_data[0].get('gene')
        else:
            ensembl_id = ensembl_data.get('gene')

        if ensembl_id:
            symbol_to_ensembl[symbol] = ensembl_id
    else:
        unmapped_symbols.append(result['query'])

print(f"  Mapped symbols: {len(symbol_to_ensembl):,}")
print(f"  Unmapped symbols: {len(unmapped_symbols):,}")

if unmapped_symbols:
    print(f"  Example unmapped symbols: {unmapped_symbols[:5]}")

# Update pert_targets with ensembl ids
trt_sh_targets['ensembl_id'] = trt_sh_targets['pert_iname'].map(symbol_to_ensembl)
print(f"\nAnnotated trt_sh targets with Ensembl IDs:")
print(f"  Total entries: {len(trt_sh_targets):,}")
print(f"  With Ensembl ID: {trt_sh_targets['ensembl_id'].notna().sum():,}")
print(f"  Without Ensembl ID: {trt_sh_targets['ensembl_id'].isna().sum():,}")

# Rename to gene_symbol and ensembl_id, drop pert_type
trt_sh_targets = trt_sh_targets.rename(columns={'pert_iname': 'gene_symbol'})
trt_sh_targets = trt_sh_targets[['pert_id', 'gene_symbol', 'ensembl_id']].copy()
print(f"\nPrepared target annotation dataframe:")
print(f"  Columns: {list(trt_sh_targets.columns)}")
print(f"  Unique pert_ids: {trt_sh_targets['pert_id'].nunique():,}")

# Load trt_sh data
trt_sh_path = os.path.join(DATA_DIR, 'trt_sh_qc.csv')
print(f"\nLoading trt_sh_qc data from: {trt_sh_path}")
trt_sh_df = pd.read_csv(trt_sh_path, low_memory=False)
print(f"Loaded trt_sh_qc data:")
print(f"  Total samples: {len(trt_sh_df):,}")
print(f"  Unique pert_ids: {trt_sh_df['pert_id'].nunique():,}")

# Merge on pert_id to annotate trt_sh_qc with gene_symbol and ensembl_id and save
print(f"\nMerging target annotations with trt_sh_qc...")
trt_sh_genes_df = trt_sh_df.merge(
    trt_sh_targets,
    on='pert_id',
    how='left'
)

print(f"Merge results:")
print(f"  Total samples: {len(trt_sh_genes_df):,}")
print(f"  Samples with gene_symbol: {trt_sh_genes_df['gene_symbol'].notna().sum():,}")
print(f"  Samples with ensembl_id: {trt_sh_genes_df['ensembl_id'].notna().sum():,}")
print(f"  Unique pert_ids: {trt_sh_genes_df['pert_id'].nunique():,}")
print(f"  Unique gene_symbols: {trt_sh_genes_df['gene_symbol'].nunique():,}")
print(f"  Unique ensembl_ids: {trt_sh_genes_df['ensembl_id'].nunique():,}")

# Save annotated dataset
output_path = os.path.join(DATA_DIR, 'trt_sh_genes_qc.csv')
print(f"\nSaving annotated data to: {output_path}")
trt_sh_genes_df.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("TRT_SH TARGET ANNOTATION COMPLETE")
print("=" * 80)
print(f"\nInput:")
print(f"  trt_sh_qc: {trt_sh_path}")
print(f"  pert_target: {pert_target_path}")
print(f"\nOutput: {output_path}")
print(f"  Total samples: {len(trt_sh_genes_df):,}")
print(f"  With target annotations: {trt_sh_genes_df['ensembl_id'].notna().sum():,}")
print(f"  Unique targets (Ensembl IDs): {trt_sh_genes_df['ensembl_id'].nunique():,}")
print("\n✓ Done")
