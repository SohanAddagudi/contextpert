#!/usr/bin/env python
"""
Drug-Target Mapping Evaluation using Metagenes from LINCS Expression Data

Evaluates drug-target interaction prediction using PCA-compressed gene expression
profiles (metagenes) from LINCS L1000 data:
- Drug representations: Metagenes from compound perturbations (trt_cp)
- Target representations: Metagenes from shRNA knockdowns (trt_sh)

Both drug and target data are transformed using a single unified PCA model
fitted on the combined expression data, ensuring they share the same latent space.
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from contextpert import submit_drug_target_mapping

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']

print("=" * 80)
print("DRUG-TARGET METAGENE EVALUATION")
print("=" * 80)
print("\nThis example uses PCA-compressed gene expression profiles (metagenes)")
print("to evaluate drug-target interaction prediction:")
print("  - Drugs: Metagenes from compound perturbations (trt_cp)")
print("  - Targets: Metagenes from shRNA knockdowns (trt_sh)")
print("  - Single unified PCA applied to both datasets")
print()

# ============================================================================
# Part 1: Load and Process Drug Data (trt_cp)
# ============================================================================
print("=" * 80)
print("LOADING DRUG DATA (COMPOUND PERTURBATIONS)")
print("=" * 80)

trt_cp_path = os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv')
print(f"\nLoading compound perturbation data from: {trt_cp_path}")
trt_cp_df = pd.read_csv(trt_cp_path)

print(f"Loaded trt_cp data:")
print(f"  Total samples: {len(trt_cp_df):,}")
print(f"  Unique BRD IDs: {trt_cp_df['pert_id'].nunique():,}")
print(f"  Unique canonical SMILES: {trt_cp_df['canonical_smiles'].nunique():,}")

# Load gene columns from shared file
gene_cols_path = os.path.join(DATA_DIR, 'trt_sh_qc_gene_cols.txt')
print(f"\nLoading gene columns from: {gene_cols_path}")
with open(gene_cols_path, 'r') as f:
    gene_cols = [line.strip() for line in f]

print(f"  Loaded {len(gene_cols)} gene features (Entrez IDs)")
print(f"  Example Entrez IDs: {gene_cols[:5]}")

# Aggregate expression by SMILES (average across replicates)
print("\nAggregating expression profiles by SMILES...")
agg_dict = {col: 'mean' for col in gene_cols}
agg_dict['canonical_smiles'] = 'first'

drug_expr_df = (
    trt_cp_df.groupby('pert_id')[gene_cols + ['canonical_smiles']]
    .agg(agg_dict)
    .reset_index()
)

print(f"  Aggregated to {len(drug_expr_df):,} unique compounds")

print(f"\nDrug expression data prepared:")
print(f"  Unique compounds: {len(drug_expr_df)}")
print(f"  Gene features (Entrez IDs): {len(gene_cols)}")
print(f"  Shape: {drug_expr_df.shape}")

# ============================================================================
# Part 2: Load and Process Target Data (trt_sh)
# ============================================================================
print("\n" + "=" * 80)
print("LOADING TARGET DATA (shRNA KNOCKDOWNS)")
print("=" * 80)

trt_sh_genes_path = os.path.join(DATA_DIR, 'trt_sh_genes_qc.csv')
print(f"\nLoading shRNA knockdown data with target annotations from: {trt_sh_genes_path}")
trt_sh_df = pd.read_csv(trt_sh_genes_path, low_memory=False)

print(f"Loaded trt_sh_genes_qc data:")
print(f"  Total samples: {len(trt_sh_df):,}")
print(f"  Unique perturbation IDs: {trt_sh_df['pert_id'].nunique():,}")
print(f"  Samples with target annotation: {trt_sh_df['ensembl_id'].notna().sum():,}")
print(f"  Unique target genes: {trt_sh_df['ensembl_id'].nunique():,}")

# Use the same gene col features (loaded earlier)
print(f"\n  Using same {len(gene_cols)} gene features for target data")

# Filter to only perturbations with target annotations
print("\nFiltering to perturbations with target annotations...")
trt_sh_df = trt_sh_df[trt_sh_df['ensembl_id'].notna()].copy()
print(f"  Retained samples: {len(trt_sh_df):,}")

# Aggregate expression by target gene (average across perturbations targeting same gene)
print("\nAggregating by target gene...")
agg_dict_sh = {col: 'mean' for col in gene_cols}

target_expr_df = (
    trt_sh_df.groupby('ensembl_id')[gene_cols]
    .mean()
    .reset_index()
)

print(f"  Aggregated to {len(target_expr_df):,} unique target genes")

target_final = target_expr_df.rename(columns={'ensembl_id': 'targetId'})

print(f"\nTarget expression data prepared:")
print(f"  Unique targets: {len(target_final)}")
print(f"  Gene features (Entrez IDs): {len(gene_cols)}")
print(f"  Shape: {target_final.shape}")

# ============================================================================
# Part 3: Apply Unified PCA to Combined Data
# ============================================================================
print("\n" + "=" * 80)
print("APPLYING UNIFIED PCA DIMENSIONALITY REDUCTION")
print("=" * 80)

# Both datasets already use the same gene columns (gene_cols)
print(f"Both drug and target data use the same {len(gene_cols)} gene features (Entrez IDs)")

# Extract expression matrices
drug_expr_matrix = drug_expr_df[gene_cols].values
target_expr_matrix = target_final[gene_cols].values

print(f"\nExpression matrices:")
print(f"  Drug matrix shape: {drug_expr_matrix.shape}")
print(f"  Target matrix shape: {target_expr_matrix.shape}")

# Combine both datasets for unified PCA
print("\nCombining drug and target data for unified PCA...")
combined_expr = np.vstack([drug_expr_matrix, target_expr_matrix])
print(f"  Combined matrix shape: {combined_expr.shape}")

# Standardize and apply PCA
n_components = 50
print(f"\nCompressing {len(gene_cols)} genes to {n_components} metagenes using PCA...")

scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined_expr)

pca = PCA(n_components=n_components, random_state=42)
combined_pca = pca.fit_transform(combined_scaled)

print(f"PCA complete!")
print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
print(f"  First 10 components explain: {pca.explained_variance_ratio_[:10].sum():.4f}")

# Split back into drug and target representations
n_drugs = len(drug_expr_df)
drug_pca = combined_pca[:n_drugs, :]
target_pca = combined_pca[n_drugs:, :]

print(f"\nSplit PCA representations:")
print(f"  Drug PCA shape: {drug_pca.shape}")
print(f"  Target PCA shape: {target_pca.shape}")

# ============================================================================
# Part 4: Prepare Final Prediction Dataframes
# ============================================================================
print("\n" + "=" * 80)
print("PREPARING PREDICTION DATAFRAMES")
print("=" * 80)

# Drug predictions: SMILES + metagene features
drug_pred_data = {'smiles': drug_expr_df['canonical_smiles'].values}
for i in range(n_components):
    drug_pred_data[f'metagene_{i}'] = drug_pca[:, i]

drug_preds = pd.DataFrame(drug_pred_data)

print(f"\nDrug predictions:")
print(f"  Unique compounds: {len(drug_preds)}")
print(f"  Representation dimensionality: {n_components} metagenes")
print(f"  Shape: {drug_preds.shape}")

# Target predictions: targetId + metagene features
target_pred_data = {'targetId': target_final['targetId'].values}
for i in range(n_components):
    target_pred_data[f'metagene_{i}'] = target_pca[:, i]

target_preds = pd.DataFrame(target_pred_data)

print(f"\nTarget predictions:")
print(f"  Unique targets: {len(target_preds)}")
print(f"  Representation dimensionality: {n_components} metagenes")
print(f"  Shape: {target_preds.shape}")

# ============================================================================
# Part 5: Run Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING DRUG-TARGET MAPPING EVALUATION")
print("=" * 80)
print("\nEvaluating using LINCS mode (default)")
print("This filters to drug-target pairs present in high-quality LINCS data\n")

results = submit_drug_target_mapping(drug_preds, target_preds, mode='lincs')

# ============================================================================
# Part 6: Summary
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

print("\nData Sources:")
print(f"  Drugs (trt_cp):   {len(drug_preds):,} compounds with {n_components} metagene features")
print(f"  Targets (trt_sh): {len(target_preds):,} genes with {n_components} metagene features")

print("\nDimensionality Reduction:")
print(f"  Original features: {len(gene_cols)} genes")
print(f"  Compressed to: {n_components} metagenes")
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
print(f"  Compression ratio: {len(gene_cols)/n_components:.1f}x")

print("\nKey Metrics:")
print(f"  AUROC:                    {results.get('auroc', 0):.4f}")
print(f"  AUPRC:                    {results.get('auprc', 0):.4f}")
print(f"  Drug->Target Precision@10: {results.get('drug_to_target_precision@10', 0):.4f}")
print(f"  Target->Drug Precision@10: {results.get('target_to_drug_precision@10', 0):.4f}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print("\nThis evaluation uses PCA-compressed gene expression profiles (metagenes)")
print("from LINCS L1000 data. A single unified PCA was applied to both drug and")
print("target expression data, ensuring they share the same latent space.")
print()
