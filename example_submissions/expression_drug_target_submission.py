#!/usr/bin/env python
"""
Drug-Target Mapping Evaluation using LINCS Expression Data

Evaluates drug-target interaction prediction using gene expression profiles from
LINCS L1000 data:
- Drug representations: Expression profiles from compound perturbations (trt_cp)
- Target representations: Expression profiles from shRNA knockdowns (trt_sh)

Both are aggregated by averaging across replicates for each unique perturbation.
"""

import os
import pandas as pd
import numpy as np

from contextpert import submit_drug_target_mapping

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']

print("=" * 80)
print("DRUG-TARGET EXPRESSION EVALUATION")
print("=" * 80)
print("\nThis example uses LINCS L1000 gene expression profiles to evaluate")
print("drug-target interaction prediction:")
print("  - Drugs: Expression from compound perturbations (trt_cp)")
print("  - Targets: Expression from shRNA knockdowns (trt_sh)")
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
drug_expr_ensembl = drug_expr_df[['canonical_smiles']].copy()
for entrez_id in mappable_genes:
    ensembl_id = entrez_to_ensembl[entrez_id]
    drug_expr_ensembl[ensembl_id] = drug_expr_df[entrez_id]

# Prepare drug prediction dataframe
drug_preds = drug_expr_ensembl.rename(columns={'canonical_smiles': 'smiles'})

print(f"\nFinal drug representation:")
print(f"  Unique compounds: {len(drug_preds)}")
print(f"  Gene features (ENSG): {len(drug_preds.columns) - 1}")
print(f"  Shape: {drug_preds.shape}")

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

# Map perturbation IDs to target ENSG IDs
# For shRNA, the pert_id represents the knockdown target gene
print("\nMapping perturbation IDs to target genes...")

# Extract target gene from pert_id (format varies, but often contains gene symbol or ID)
# For now, we'll use a simple heuristic: aggregate by pert_id and use the most downregulated gene
# as the inferred target

print("  Aggregating expression profiles by perturbation ID...")
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

# Aggregate by inferred target (average expression across perturbations targeting same gene)
print("\nAggregating by target gene...")
target_final = (
    target_expr_df.groupby('inferred_target')[gene_cols_ensembl]
    .mean()
    .reset_index()
)

# Prepare target prediction dataframe
target_preds = target_final.rename(columns={'inferred_target': 'targetId'})

print(f"\nFinal target representation:")
print(f"  Unique targets: {len(target_preds)}")
print(f"  Gene features (ENSG): {len(target_preds.columns) - 1}")
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
print(f"  Drugs (trt_cp):   {len(drug_preds):,} compounds with {len(drug_preds.columns)-1} gene features")
print(f"  Targets (trt_sh): {len(target_preds):,} genes with {len(target_preds.columns)-1} gene features")

print("\nKey Metrics:")
print(f"  AUROC:                    {results.get('auroc', 0):.4f}")
print(f"  AUPRC:                    {results.get('auprc', 0):.4f}")
print(f"  Drug->Target Precision@10: {results.get('drug_to_target_precision@10', 0):.4f}")
print(f"  Target->Drug Precision@10: {results.get('target_to_drug_precision@10', 0):.4f}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print("\nThis evaluation uses real LINCS L1000 gene expression profiles to assess")
print("whether expression-based representations can predict known drug-target interactions.")
print()
