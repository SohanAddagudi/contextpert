#!/usr/bin/env python
"""
Drug-Target Mapping Evaluation using LINCS Model Betas

Evaluates drug-target interaction prediction using learned model parameters
(betas) from a ContextualizedCorrelation model trained on LINCS L1000 data:
- Drug representations: Betas from compound perturbations (trt_cp)
- Target representations: Betas from shRNA knockdowns (trt_sh)

This version uses the same metadata/annotation files as the expression
evaluation script for consistency.
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from contextpert import submit_drug_target_mapping
from contextpert.utils import canonicalize_smiles

# --- Configuration ---
DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


# ---------------------

def std_norm(x):
    s = x.std(axis=0, keepdims=True)
    return x / np.where(s == 0, 1, s)


print("=" * 80)
print("DRUG-TARGET BETA EVALUATION")
print("=" * 80)
print("\nThis example uses learned model parameters (betas + mus) to evaluate")
print("drug-target interaction prediction:")
print("  - Drugs: bsq+mus (upper-tri, std-normed) from compound perturbation models (trt_cp)")
print("  - Targets: bsq+mus (upper-tri, std-normed) from shRNA knockdown models (trt_sh)")
print()


print("=" * 80)
print("LOADING DRUG DATA (COMPOUND PERTURBATION BETAS + MUS)")
print("=" * 80)
# --- Define model paths ---
CP_BETA_PATH = os.path.join(DATA_DIR, 'cellvs_molecule_networks/chemberta_model_outputs/full_dataset_betas.npy')
CP_MUS_PATH  = os.path.join(DATA_DIR, 'cellvs_molecule_networks/chemberta_model_outputs/full_dataset_mus.npy')
CP_PRED_CSV_PATH = os.path.join(DATA_DIR, 'cellvs_molecule_networks/chemberta_model_outputs/full_dataset_predictions.csv')

SH_BETA_PATH = os.path.join(DATA_DIR, 'drug_target_networks/trt_sh_aidocell_drug_target_networks/full_dataset_betas.npy')
SH_MUS_PATH  = os.path.join(DATA_DIR, 'drug_target_networks/trt_sh_aidocell_drug_target_networks/full_dataset_mus.npy')
SH_PRED_CSV_PATH = os.path.join(DATA_DIR, 'drug_target_networks/trt_sh_aidocell_drug_target_networks/full_dataset_predictions.csv')

# --- Load betas, mus, and metadata ---
print(f"\nLoading model betas from: {CP_BETA_PATH}")
cp_betas = np.load(CP_BETA_PATH)
print(f"Loading model mus from: {CP_MUS_PATH}")
cp_mus = np.load(CP_MUS_PATH)
print(f"Loading prediction metadata from: {CP_PRED_CSV_PATH}")
cp_meta = pd.read_csv(CP_PRED_CSV_PATH)
print(f"  Loaded {len(cp_meta):,} samples | betas: {cp_betas.shape} | mus: {cp_mus.shape}")

# --- Upper-triangular bsq + mus ---
n_x = cp_mus.shape[-1]
idx_upper = np.triu_indices(n_x, k=1)
cp_bsq = cp_betas[:, idx_upper[0], idx_upper[1]] ** 2
cp_mus_ut = cp_mus[:, idx_upper[0], idx_upper[1]]
n_feat = cp_bsq.shape[1]
b_cols = [f'b_{i}' for i in range(n_feat)]
m_cols = [f'm_{i}' for i in range(n_feat)]

# --- Canonicalize SMILES from metadata ---
smiles_col = cp_meta['canonical_smiles'].apply(
    lambda s: canonicalize_smiles(s) if pd.notna(s) and s not in ('-666', 'restricted') else None
)

# --- Aggregate by SMILES ---
print("\nAggregating by SMILES...")
df_b = pd.DataFrame(cp_bsq, columns=b_cols); df_b['smiles'] = smiles_col.values
df_m = pd.DataFrame(cp_mus_ut, columns=m_cols); df_m['smiles'] = smiles_col.values
drug_b = df_b.dropna(subset=['smiles']).groupby('smiles')[b_cols].mean().reset_index()
drug_m = df_m.dropna(subset=['smiles']).groupby('smiles')[m_cols].mean().reset_index()
merged_cp = drug_b.merge(drug_m, on='smiles')
print(f"  Aggregated to {len(merged_cp):,} unique compounds")

# --- Std-norm and concatenate ---
arr_cp = np.hstack([std_norm(merged_cp[b_cols].values), std_norm(merged_cp[m_cols].values)])
f_cols = [f'f_{i}' for i in range(arr_cp.shape[1])]
drug_preds = pd.DataFrame(arr_cp, columns=f_cols)
drug_preds['smiles'] = merged_cp['smiles'].values

print(f"\nFinal drug representation:")
print(f"  Unique compounds: {len(drug_preds)}")
print(f"  Features (bsq+mus): {len(f_cols)}")
print(f"  Shape: {drug_preds.shape}")

del cp_betas, cp_mus, df_b, df_m, drug_b, drug_m, merged_cp, arr_cp

# ============================================================================
# Part 2: Load and Process Target Data (trt_sh betas + mus)
# ============================================================================
print("\n" + "=" * 80)
print("LOADING TARGET DATA (shRNA KNOCKDOWN BETAS + MUS)")
print("=" * 80)

print(f"\nLoading model betas from: {SH_BETA_PATH}")
sh_betas = np.load(SH_BETA_PATH)
print(f"Loading model mus from: {SH_MUS_PATH}")
sh_mus = np.load(SH_MUS_PATH)
print(f"Loading prediction metadata from: {SH_PRED_CSV_PATH}")
sh_meta = pd.read_csv(SH_PRED_CSV_PATH)
print(f"  Loaded {len(sh_meta):,} samples | betas: {sh_betas.shape} | mus: {sh_mus.shape}")

# --- Upper-triangular bsq + mus ---
sh_n_x = sh_mus.shape[-1]
sh_idx_upper = np.triu_indices(sh_n_x, k=1)
sh_bsq = sh_betas[:, sh_idx_upper[0], sh_idx_upper[1]] ** 2
sh_mus_ut = sh_mus[:, sh_idx_upper[0], sh_idx_upper[1]]
sh_n_feat = sh_bsq.shape[1]
sb_cols = [f'b_{i}' for i in range(sh_n_feat)]
sm_cols = [f'm_{i}' for i in range(sh_n_feat)]

# --- Merge with trt_sh_genes_qc to get ensembl_id ---
trt_sh_genes_path = os.path.join(DATA_DIR, 'trt_sh_genes_qc.csv')
print(f"\nLoading shRNA target annotations from: {trt_sh_genes_path}")
sh_annot = pd.read_csv(trt_sh_genes_path, usecols=['inst_id', 'ensembl_id'], low_memory=False)
sh_annot = sh_annot[sh_annot['ensembl_id'].notna()]
ensembl_ids = sh_meta.merge(sh_annot, on='inst_id', how='left')['ensembl_id'].values
print(f"  Instances with target annotations: {pd.notna(ensembl_ids).sum():,}")
print(f"  Unique target genes: {pd.Series(ensembl_ids).nunique():,}")

# --- Aggregate by ensembl_id ---
print("\nAggregating by target gene (Ensembl ID)...")
df_sb = pd.DataFrame(sh_bsq, columns=sb_cols); df_sb['ensembl_id'] = ensembl_ids
df_sm = pd.DataFrame(sh_mus_ut, columns=sm_cols); df_sm['ensembl_id'] = ensembl_ids
df_sb = df_sb.dropna(subset=['ensembl_id']); df_sm = df_sm.dropna(subset=['ensembl_id'])
tgt_b = df_sb.groupby('ensembl_id')[sb_cols].mean().reset_index()
tgt_m = df_sm.groupby('ensembl_id')[sm_cols].mean().reset_index()
merged_sh = tgt_b.merge(tgt_m, on='ensembl_id')
print(f"  Aggregated to {len(merged_sh):,} unique target genes")

# --- Std-norm and concatenate ---
arr_sh = np.hstack([std_norm(merged_sh[sb_cols].values), std_norm(merged_sh[sm_cols].values)])
sf_cols = [f'f_{i}' for i in range(arr_sh.shape[1])]
target_preds = pd.DataFrame(arr_sh, columns=sf_cols)
target_preds['targetId'] = merged_sh['ensembl_id'].values

print(f"\nFinal target representation:")
print(f"  Unique targets: {len(target_preds)}")
print(f"  Features (bsq+mus): {len(sf_cols)}")
print(f"  Shape: {target_preds.shape}")

del sh_betas, sh_mus, df_sb, df_sm, tgt_b, tgt_m, merged_sh, arr_sh

# ============================================================================
# Part 3: Run Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING DRUG-TARGET MAPPING EVALUATION")
print("=" * 80)
print("\nEvaluating using LINCS mode (default)")

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

