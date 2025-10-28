#!/usr/bin/env python
"""
Drug-Target Mapping Evaluation using AIDO Cell Embeddings

Evaluates drug-target interaction prediction using cell embeddings from
AIDO Cell 3M model:
- Drug representations: Embeddings from compound perturbations (trt_cp)
- Target representations: Embeddings from shRNA knockdowns (trt_sh)

Both are aggregated by averaging across replicates for each unique perturbation.
"""

import os
import pandas as pd
import numpy as np

from contextpert import submit_drug_target_mapping

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']

print("=" * 80)
print("DRUG-TARGET AIDO CELL 3M EMBEDDING EVALUATION")
print("=" * 80)
print("\nThis example uses AIDO Cell 3M embeddings to evaluate")
print("drug-target interaction prediction:")
print("  - Drugs: Embeddings from compound perturbations (trt_cp)")
print("  - Targets: Embeddings from shRNA knockdowns (trt_sh)")
print()

# ============================================================================
# Part 1: Load and Process Drug Data (trt_cp)
# ============================================================================
print("=" * 80)
print("LOADING DRUG DATA (COMPOUND PERTURBATIONS)")
print("=" * 80)

trt_cp_embed_path = os.path.join(DATA_DIR, 'trt_cp_smiles_qc_aido_cell_3m_embeddings.csv')
print(f"\nLoading compound perturbation embeddings from: {trt_cp_embed_path}")
trt_cp_df = pd.read_csv(trt_cp_embed_path)

print(f"Loaded trt_cp embedding data:")
print(f"  Total samples: {len(trt_cp_df):,}")
print(f"  Unique BRD IDs: {trt_cp_df['pert_id'].nunique():,}")
print(f"  Unique canonical SMILES: {trt_cp_df['canonical_smiles'].nunique():,}")

# Identify embedding columns
metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                 'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                 'pct_self_rank_q25', 'canonical_smiles', 'inchi_key']
embedding_cols = [col for col in trt_cp_df.columns if col.startswith('emb_')]

print(f"  Embedding features: {len(embedding_cols)}")
print(f"  Example embedding columns: {embedding_cols[:5]}")

# Aggregate embeddings by SMILES (average across replicates)
print("\nAggregating embeddings by SMILES...")
agg_dict = {col: 'mean' for col in embedding_cols}
agg_dict['canonical_smiles'] = 'first'

drug_embed_df = (
    trt_cp_df.groupby('pert_id')[embedding_cols + ['canonical_smiles']]
    .agg(agg_dict)
    .reset_index()
)

print(f"  Aggregated to {len(drug_embed_df):,} unique compounds")

# Prepare drug prediction dataframe
drug_preds = drug_embed_df[['canonical_smiles'] + embedding_cols].rename(columns={'canonical_smiles': 'smiles'})

print(f"\nFinal drug representation:")
print(f"  Unique compounds: {len(drug_preds)}")
print(f"  Embedding features: {len(drug_preds.columns) - 1}")
print(f"  Shape: {drug_preds.shape}")

# ============================================================================
# Part 2: Load and Process Target Data (trt_sh)
# ============================================================================
print("\n" + "=" * 80)
print("LOADING TARGET DATA (shRNA KNOCKDOWNS)")
print("=" * 80)

trt_sh_embed_path = os.path.join(DATA_DIR, 'trt_sh_genes_qc_aido_cell_3m_embeddings.csv')
print(f"\nLoading shRNA knockdown embeddings with target annotations from: {trt_sh_embed_path}")
trt_sh_df = pd.read_csv(trt_sh_embed_path, low_memory=False)

print(f"Loaded trt_sh embedding data:")
print(f"  Total samples: {len(trt_sh_df):,}")
print(f"  Unique perturbation IDs: {trt_sh_df['pert_id'].nunique():,}")
print(f"  Samples with target annotation: {trt_sh_df['ensembl_id'].notna().sum():,}")
print(f"  Unique target genes: {trt_sh_df['ensembl_id'].nunique():,}")

# Identify embedding columns (same as drug data)
print(f"\n  Using {len(embedding_cols)} embedding features")

# Filter to only perturbations with target annotations
print("\nFiltering to perturbations with target annotations...")
trt_sh_df = trt_sh_df[trt_sh_df['ensembl_id'].notna()].copy()
print(f"  Retained samples: {len(trt_sh_df):,}")

# Aggregate embeddings by target gene (average across perturbations targeting same gene)
print("\nAggregating by target gene...")
agg_dict_sh = {col: 'mean' for col in embedding_cols}

target_embed_df = (
    trt_sh_df.groupby('ensembl_id')[embedding_cols]
    .mean()
    .reset_index()
)

print(f"  Aggregated to {len(target_embed_df):,} unique target genes")

# Prepare target prediction dataframe (targetId + same embedding_cols as drug data)
target_preds = target_embed_df.rename(columns={'ensembl_id': 'targetId'})

print(f"\nFinal target representation:")
print(f"  Unique targets: {len(target_preds)}")
print(f"  Embedding features: {len(target_preds.columns) - 1}")
print(f"  Shape: {target_preds.shape}")

# ============================================================================
# Part 3: Run Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING DRUG-TARGET MAPPING EVALUATION")
print("=" * 80)
print("\nEvaluating using LINCS mode (default)")
print("This filters to drug-target pairs present in high-quality LINCS data\n")

results = submit_drug_target_mapping(drug_preds, target_preds, mode='lincs')

# ============================================================================
# Part 4: Summary
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

print("\nData Sources:")
print(f"  Drugs (trt_cp):   {len(drug_preds):,} compounds with {len(drug_preds.columns)-1} embedding features")
print(f"  Targets (trt_sh): {len(target_preds):,} genes with {len(target_preds.columns)-1} embedding features")

print("\nKey Metrics:")
print(f"  AUROC:                    {results.get('auroc', 0):.4f}")
print(f"  AUPRC:                    {results.get('auprc', 0):.4f}")
print(f"  Drug->Target Precision@10: {results.get('drug_to_target_precision@10', 0):.4f}")
print(f"  Target->Drug Precision@10: {results.get('target_to_drug_precision@10', 0):.4f}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print("\nThis evaluation uses AIDO Cell 3M embeddings (128-dim) to assess")
print("whether learned cell state representations can predict known drug-target interactions.")
print()
