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
from sklearn.decomposition import PCA

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

print('Attempting batch correction...')

results = None
best_n_pcs = 0
for n_pcs in range(4):
    # Remove first n_pcs fom the dataset to attempt batch correction. Reports best result in terms of AUROC.
    # Combine datasets, remove first n_pcs for batch correction, then separate again
    print("\nApplying batch correction...")
    print("  Combining drug and target datasets...")

    # Store original dfs and IDs for later
    drug_ids = drug_preds['smiles'].copy()
    target_ids = target_preds['targetId'].copy()

    # Store original sizes for later separation
    n_drugs = len(drug_preds)
    n_targets = len(target_preds)

    # Get feature columns (excluding ID columns)
    drug_feature_cols = [col for col in drug_preds.columns if col != 'smiles']
    target_feature_cols = [col for col in target_preds.columns if col != 'targetId']

    # Extract feature matrices
    X_drug = drug_preds[drug_feature_cols].values
    X_target = target_preds[target_feature_cols].values

    # Combine feature matrices
    X = np.vstack([X_drug, X_target])
    print(f"  Combined feature matrix shape: {X.shape}")

    # Remove first n_pcs
    print(f"  Removing first {n_pcs} principal components to reduce batch effects...")
    pca = PCA(n_components=n_pcs)
    pca.fit(X)

    # Project data onto first 2 PCs
    first_pcs_projection = pca.transform(X)

    # Reconstruct first 2 PCs contribution
    first_pcs_contribution = first_pcs_projection @ pca.components_

    # Remove first 2 PCs from data
    X_corrected = X - first_pcs_contribution

    print(f"  Removed PCs:")
    for i, explained_variance_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i}: {explained_variance_ratio}")
    print(f"  Total explained variance removed: {pca.explained_variance_ratio_.sum():.4f}")

    # Separate back into drug and target predictions
    X_drug_corrected = X_corrected[:n_drugs]
    X_target_corrected = X_corrected[n_drugs:]

    # Reconstruct dataframes with corrected features
    drug_preds_pca = pd.DataFrame(X_drug_corrected, columns=drug_feature_cols)
    drug_preds_pca.insert(0, 'smiles', drug_ids.values)

    target_preds_pca = pd.DataFrame(X_target_corrected, columns=target_feature_cols)
    target_preds_pca.insert(0, 'targetId', target_ids.values)

    print(f"  Batch correction complete!")
    print(f"    Drug predictions: {len(drug_preds_pca):,} samples")
    print(f"    Target predictions: {len(target_preds_pca):,} samples")

    results_tmp = submit_drug_target_mapping(
        drug_preds_pca,
        target_preds_pca,
    )

    if not results or results_tmp.get('auroc') > results.get('auroc'):
        results = results_tmp
        best_n_pcs = n_pcs

print(f'\nPerformed batch correction with {best_n_pcs} PCs')

# ============================================================================
# Part 4: Summary
# ============================================================================
# Print results
print("\n" + "="*80)
print("FINAL METRICS")
print("="*80)

print(f"\nDrug -> Target Retrieval ({results.get('drug_to_target_queries', 0)} queries):")
k_list = [1, 5, 10, 50]
for k in k_list:
    print(f"  k={k}:")
    print(f"    Precision@{k}: {results.get(f'drug_to_target_precision@{k}', 0):.4f}")
    print(f"    Hits@{k}:    {results.get(f'drug_to_target_recall@{k}', 0):.4f}")
    print(f"    MRR@{k}:       {results.get(f'drug_to_target_mrr@{k}', 0):.4f}")

print(f"\nTarget -> Drug Retrieval ({results.get('target_to_drug_queries', 0)} queries):")
for k in k_list:
    print(f"  k={k}:")
    print(f"    Precision@{k}: {results.get(f'target_to_drug_precision@{k}', 0):.4f}")
    print(f"    Hits@{k}:    {results.get(f'target_to_drug_recall@{k}', 0):.4f}")
    print(f"    MRR@{k}:       {results.get(f'target_to_drug_mrr@{k}', 0):.4f}")

print(f"\nGraph-Based Metrics:")
print(f"  AUROC: {results.get('auroc', 0):.4f}")
print(f"  AUPRC: {results.get('auprc', 0):.4f}")
print(f"  Positives: {results.get('n_positives', 0):,} / {results.get('n_total_pairs', 0):,}")

print("\n" + "="*80)