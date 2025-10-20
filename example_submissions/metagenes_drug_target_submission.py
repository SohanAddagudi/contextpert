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

# Identify gene expression columns (Entrez IDs)
metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                 'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                 'pct_self_rank_q25', 'canonical_smiles', 'inchi_key']
gene_cols_entrez = [col for col in trt_cp_df.columns if col not in metadata_cols]

print(f"  Gene expression features (Entrez IDs): {len(gene_cols_entrez)}")

# Map Entrez IDs to ENSG IDs for consistency with target data
print("\nMapping Entrez IDs to Ensembl IDs...")
entrez_to_ensembl_path = os.path.join(DATA_DIR, 'entrez_to_ensembl_map.csv')
mapping_df = pd.read_csv(entrez_to_ensembl_path)

# Create mapping dictionary
entrez_to_ensembl = dict(zip(
    mapping_df['entrez_id'].astype(str),
    mapping_df['ensembl_gene_id']
))

# Identify mappable genes
mappable_genes = [col for col in gene_cols_entrez if col in entrez_to_ensembl]
print(f"  Mappable genes: {len(mappable_genes)} / {len(gene_cols_entrez)} ({len(mappable_genes)/len(gene_cols_entrez)*100:.1f}%)")

# Aggregate expression by SMILES (average across replicates)
print("\nAggregating expression profiles by SMILES...")
agg_dict = {col: 'mean' for col in mappable_genes}
agg_dict['canonical_smiles'] = 'first'

drug_expr_df = (
    trt_cp_df.groupby('pert_id')[mappable_genes + ['canonical_smiles']]
    .agg(agg_dict)
    .reset_index()
)

print(f"  Aggregated to {len(drug_expr_df):,} unique compounds")

# Rename gene columns to ENSG IDs
print("\nRenaming gene columns to Ensembl IDs...")
drug_expr_ensembl = drug_expr_df[['pert_id', 'canonical_smiles']].copy()
for entrez_id in mappable_genes:
    ensembl_id = entrez_to_ensembl[entrez_id]
    drug_expr_ensembl[ensembl_id] = drug_expr_df[entrez_id]

print(f"\nDrug expression data prepared:")
print(f"  Unique compounds: {len(drug_expr_ensembl)}")
print(f"  Gene features (ENSG): {len(drug_expr_ensembl.columns) - 2}")
print(f"  Shape: {drug_expr_ensembl.shape}")

# ============================================================================
# Part 2: Load and Process Target Data (trt_sh)
# ============================================================================
print("\n" + "=" * 80)
print("LOADING TARGET DATA (shRNA KNOCKDOWNS)")
print("=" * 80)

trt_sh_path = os.path.join(DATA_DIR, 'trt_sh_qc.csv')
print(f"\nLoading shRNA knockdown data from: {trt_sh_path}")
trt_sh_df = pd.read_csv(trt_sh_path)

print(f"Loaded trt_sh data:")
print(f"  Total samples: {len(trt_sh_df):,}")
print(f"  Unique perturbation IDs: {trt_sh_df['pert_id'].nunique():,}")

# Identify gene columns (already ENSG IDs)
sh_metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                    'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                    'pct_self_rank_q25']
gene_cols_ensembl = [col for col in trt_sh_df.columns if col not in sh_metadata_cols]

print(f"  Gene expression features (ENSG IDs): {len(gene_cols_ensembl)}")

# Aggregate by perturbation ID
print("\nAggregating expression profiles by perturbation ID...")
agg_dict_sh = {col: 'mean' for col in gene_cols_ensembl}

target_expr_df = (
    trt_sh_df.groupby('pert_id')[gene_cols_ensembl]
    .agg(agg_dict_sh)
    .reset_index()
)

print(f"  Aggregated to {len(target_expr_df):,} unique perturbations")

# Infer target gene: use the most downregulated gene as the target
print("\nInferring target genes from expression profiles...")
print("  Strategy: Most downregulated gene per perturbation")

target_genes = []
for idx, row in target_expr_df.iterrows():
    expr_values = row[gene_cols_ensembl].values
    min_expr_idx = np.argmin(expr_values)
    target_gene = gene_cols_ensembl[min_expr_idx]
    target_genes.append(target_gene)

target_expr_df['inferred_target'] = target_genes

print(f"  Inferred targets for {len(target_expr_df)} perturbations")
print(f"  Unique target genes: {target_expr_df['inferred_target'].nunique()}")

# Aggregate by inferred target
print("\nAggregating by target gene...")
target_final = (
    target_expr_df.groupby('inferred_target')[gene_cols_ensembl]
    .mean()
    .reset_index()
)

target_final = target_final.rename(columns={'inferred_target': 'targetId'})

print(f"\nTarget expression data prepared:")
print(f"  Unique targets: {len(target_final)}")
print(f"  Gene features (ENSG): {len(target_final.columns) - 1}")
print(f"  Shape: {target_final.shape}")

# ============================================================================
# Part 3: Apply Unified PCA to Combined Data
# ============================================================================
print("\n" + "=" * 80)
print("APPLYING UNIFIED PCA DIMENSIONALITY REDUCTION")
print("=" * 80)

# Find common genes between drug and target data
drug_genes = set(drug_expr_ensembl.columns) - {'pert_id', 'canonical_smiles'}
target_genes_set = set(target_final.columns) - {'targetId'}
common_genes = sorted(list(drug_genes & target_genes_set))

print(f"Finding common genes:")
print(f"  Drug genes (ENSG): {len(drug_genes)}")
print(f"  Target genes (ENSG): {len(target_genes_set)}")
print(f"  Common genes: {len(common_genes)}")

# Extract expression matrices with common genes only
drug_expr_matrix = drug_expr_ensembl[common_genes].values
target_expr_matrix = target_final[common_genes].values

print(f"\nExpression matrices:")
print(f"  Drug matrix shape: {drug_expr_matrix.shape}")
print(f"  Target matrix shape: {target_expr_matrix.shape}")

# Combine both datasets for unified PCA
print("\nCombining drug and target data for unified PCA...")
combined_expr = np.vstack([drug_expr_matrix, target_expr_matrix])
print(f"  Combined matrix shape: {combined_expr.shape}")

# Standardize and apply PCA
n_components = 50
print(f"\nCompressing {len(common_genes)} genes to {n_components} metagenes using PCA...")

scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined_expr)

pca = PCA(n_components=n_components, random_state=42)
combined_pca = pca.fit_transform(combined_scaled)

print(f"PCA complete!")
print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
print(f"  First 10 components explain: {pca.explained_variance_ratio_[:10].sum():.4f}")

# Split back into drug and target representations
n_drugs = len(drug_expr_ensembl)
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
drug_pred_data = {'smiles': drug_expr_ensembl['canonical_smiles'].values}
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
print(f"  Original features: {len(common_genes)} genes")
print(f"  Compressed to: {n_components} metagenes")
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
print(f"  Compression ratio: {len(common_genes)/n_components:.1f}x")

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
