#!/usr/bin/env python
"""
Drug-Target Mapping Evaluation using LINCS Model Betas

Evaluates drug-target interaction prediction using learned model parameters
(betas) from a ContextualizedCorrelation model trained on LINCS L1000 data.
"""

import os
import pandas as pd
import numpy as np

from contextpert import submit_drug_target_mapping
from contextpert.utils import canonicalize_smiles

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']
TRAINING_OUTPUT_PARENT_DIR = '/home/user/screening2/contextpert/all_run_results'

# ============================================================================
# Part 1: Load and Process Drug Data (trt_cp Model Betas)
# ============================================================================
print("=" * 80)
print("LOADING DRUG DATA (COMPOUND PERTURBATION BETAS)")
print("=" * 80)

CP_PERT_NAME = 'trt_cp'
CP_EMB_NAME = 'AIDOcell'
CP_PARAM_NAME = 'bs_64_arch_10'

CP_MODEL_RESULTS_DIR = os.path.join(TRAINING_OUTPUT_PARENT_DIR, f'{CP_PERT_NAME}_{CP_EMB_NAME}', CP_PARAM_NAME)
CP_BETA_PATH = os.path.join(CP_MODEL_RESULTS_DIR, 'full_dataset_betas.npy')
CP_PRED_CSV_PATH = os.path.join(CP_MODEL_RESULTS_DIR, 'full_dataset_predictions.csv')

print(f"\nLoading model betas from: {CP_BETA_PATH}")
cp_betas = np.load(CP_BETA_PATH)

print(f"Loading prediction metadata from: {CP_PRED_CSV_PATH}")
cp_pred_meta_df = pd.read_csv(CP_PRED_CSV_PATH, usecols=['inst_id'])

# Flatten betas
n_samples, n_pcs, _ = cp_betas.shape
n_features = n_pcs * n_pcs
print(f"  Loaded betas for {n_samples:,} samples")
print(f"  Flattening betas into feature vectors of length {n_features:,}...")
cp_betas_flat = cp_betas.reshape(n_samples, n_features)
feature_cols = [f'beta_{i}' for i in range(n_features)]
cp_beta_df = pd.DataFrame(cp_betas_flat, columns=feature_cols)
cp_model_data_df = pd.concat([cp_pred_meta_df[['inst_id']], cp_beta_df], axis=1)

# Load LINCS metadata to map inst_id to SMILES
trt_cp_path = os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv')
print(f"\nLoading LINCS metadata from: {trt_cp_path}")
lincs_meta_df = pd.read_csv(trt_cp_path, usecols=['inst_id', 'pert_id', 'canonical_smiles'])

# Merge and filter
merged_df = pd.merge(cp_model_data_df, lincs_meta_df, on='inst_id', how='left')
bad_smiles = ['-666', 'restricted']
merged_df = merged_df[~merged_df['canonical_smiles'].isin(bad_smiles)].copy()
merged_df = merged_df[merged_df['canonical_smiles'].notna()].copy()
print(f"  Remaining samples with valid SMILES: {len(merged_df):,}")

# Aggregate by BRD ID
print("\nAggregating model (beta) representations by BRD ID...")
agg_cols = feature_cols + ['canonical_smiles']
model_rep_by_brd = (
    merged_df.groupby('pert_id')[agg_cols]
    .agg({**{col: 'mean' for col in feature_cols}, 'canonical_smiles': 'first'})
    .reset_index()
)

# Canonicalize SMILES
print("\nCanonicalizing SMILES...")
def safe_canonicalize(smiles):
    try:
        return canonicalize_smiles(smiles)
    except:
        return None

model_rep_by_brd['smiles'] = model_rep_by_brd['canonical_smiles'].apply(safe_canonicalize)
model_rep_by_brd = model_rep_by_brd[model_rep_by_brd['smiles'].notna()].copy()

drug_preds = model_rep_by_brd[['smiles'] + feature_cols].copy()
print(f"\nFinal drug representation:")
print(f"  Unique compounds: {len(drug_preds)}")
print(f"  Gene features (betas): {len(feature_cols)}")

# ============================================================================
# Part 2: Load Gene ID Mappings (Following Reference Code Exactly)
# ============================================================================
print("\n" + "=" * 80)
print("LOADING GENE ID MAPPINGS")
print("=" * 80)

# Map Entrez IDs to Ensembl IDs (exactly as in reference code)
print("\nMapping Entrez IDs to Ensembl IDs...")
entrez_to_ensembl_path = os.path.join(DATA_DIR, 'entrez_to_ensembl_map.csv')
mapping_df = pd.read_csv(entrez_to_ensembl_path)

# Create mapping dictionary (exactly as reference code)
entrez_to_ensembl = dict(zip(
    mapping_df['entrez_id'].astype(str),
    mapping_df['ensembl_id']
))

print(f"  Created mapping for {len(entrez_to_ensembl):,} Entrez IDs to Ensembl IDs")
print(f"\n  Sample mappings:")
for i, (entrez, ensembl) in enumerate(list(entrez_to_ensembl.items())[:5]):
    print(f"    {entrez} -> {ensembl}")

# ============================================================================
# Part 3: Load and Process Target Data (trt_sh Model Betas)
# ============================================================================
print("\n" + "=" * 80)
print("LOADING TARGET DATA (shRNA KNOCKDOWN BETAS)")
print("=" * 80)

trt_sh_path = os.path.join(DATA_DIR, 'trt_sh_qc.csv')

# Step 3a: Load expression data to infer target genes
print("\nLoading shRNA knockdown data from: {trt_sh_path}")
trt_sh_df = pd.read_csv(trt_sh_path)

print(f"Loaded trt_sh data:")
print(f"  Total samples: {len(trt_sh_df):,}")
print(f"  Unique perturbation IDs: {trt_sh_df['pert_id'].nunique():,}")

# Identify gene columns (Entrez IDs) - exactly as reference code
sh_metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                    'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                    'pct_self_rank_q25']
gene_cols_entrez = [col for col in trt_sh_df.columns if col not in sh_metadata_cols]

print(f"  Gene expression features (Entrez IDs): {len(gene_cols_entrez)}")

# Aggregate expression by perturbation ID - exactly as reference code
print("\n  Aggregating expression profiles by perturbation ID...")
agg_dict_sh = {col: 'mean' for col in gene_cols_entrez}

target_expr_df = (
    trt_sh_df.groupby('pert_id')[gene_cols_entrez]
    .agg(agg_dict_sh)
    .reset_index()
)

print(f"  Aggregated to {len(target_expr_df):,} unique perturbations")

# Infer target gene - exactly as reference code
print("\nInferring target genes from expression profiles...")
print("  Strategy: Most downregulated gene per perturbation")

target_genes = []
for idx, row in target_expr_df.iterrows():
    expr_values = row[gene_cols_entrez].values
    min_expr_idx = np.argmin(expr_values)
    target_gene = gene_cols_entrez[min_expr_idx]  # This is Entrez ID
    target_genes.append(target_gene)

target_expr_df['inferred_target'] = target_genes

print(f"  Inferred targets for {len(target_expr_df)} perturbations")
print(f"  Unique target genes: {target_expr_df['inferred_target'].nunique()}")

# THE KEY STEP: Convert Entrez IDs to Ensembl IDs (exactly as reference code does)
print("\nConverting target Entrez IDs to Ensembl IDs...")
target_expr_df['inferred_target_ensembl'] = target_expr_df['inferred_target'].map(entrez_to_ensembl)

# Filter out unmapped targets
before_filter = len(target_expr_df)
target_expr_df = target_expr_df[target_expr_df['inferred_target_ensembl'].notna()].copy()
after_filter = len(target_expr_df)
print(f"  Successfully mapped {after_filter}/{before_filter} targets to Ensembl IDs")
print(f"  Unique Ensembl target IDs: {target_expr_df['inferred_target_ensembl'].nunique()}")

# Create pert_id -> Ensembl target mapping
target_map_df = target_expr_df[['pert_id', 'inferred_target_ensembl']].drop_duplicates()

print(f"\n  Sample target Ensembl IDs (first 10):")
for i, target in enumerate(target_map_df['inferred_target_ensembl'].head(10)):
    print(f"    {i+1}. {target}")

# Clear expression data from memory
del trt_sh_df, target_expr_df

# Step 3b: Load trt_sh model betas
print("\nLoading trt_sh model betas...")
TRAINING_OUTPUT_PARENT_SH = '/home/user/screening2/contextpert/prev_inferences'
SH_MODEL_RESULTS_DIR = os.path.join(TRAINING_OUTPUT_PARENT_SH, 'trt_sh_AIDOprot_struct')
SH_BETA_PATH = os.path.join(SH_MODEL_RESULTS_DIR, 'full_dataset_betas.npy')
SH_PRED_CSV_PATH = os.path.join(SH_MODEL_RESULTS_DIR, 'full_dataset_predictions.csv')

print(f"  Loading from: {SH_BETA_PATH}")
sh_betas = np.load(SH_BETA_PATH)

print(f"  Loading metadata from: {SH_PRED_CSV_PATH}")
sh_pred_meta_df = pd.read_csv(SH_PRED_CSV_PATH, usecols=['inst_id'])

# Flatten betas
sh_n_samples, sh_n_pcs, _ = sh_betas.shape
sh_n_features = sh_n_pcs * sh_n_pcs
print(f"  Loaded betas for {sh_n_samples:,} samples")
sh_betas_flat = sh_betas.reshape(sh_n_samples, sh_n_features)
sh_feature_cols = [f'beta_{i}' for i in range(sh_n_features)]
sh_beta_df = pd.DataFrame(sh_betas_flat, columns=sh_feature_cols)
sh_model_data_df = pd.concat([sh_pred_meta_df[['inst_id']], sh_beta_df], axis=1)

# Step 3c: Merge betas with metadata and target mapping
print("\nMerging betas with instance metadata (to get pert_id)...")
sh_meta_df = pd.read_csv(trt_sh_path, usecols=['inst_id', 'pert_id'])
sh_merged_df = pd.merge(sh_model_data_df, sh_meta_df, on='inst_id', how='left')

print("Merging betas with pert_id -> Ensembl target map...")
sh_merged_df = pd.merge(sh_merged_df, target_map_df, on='pert_id', how='inner')
print(f"  Matched {len(sh_merged_df):,} instances to target genes")

# Aggregate by inferred target Ensembl ID (exactly as reference code)
print("\nAggregating betas by target gene...")
target_final = (
    sh_merged_df.groupby('inferred_target_ensembl')[sh_feature_cols]
    .mean()
    .reset_index()
)

# Prepare target prediction dataframe (exactly as reference code)
target_preds = target_final.rename(columns={'inferred_target_ensembl': 'targetId'})

print(f"\nFinal target representation:")
print(f"  Unique targets: {len(target_preds)}")
print(f"  Gene features (betas): {len(target_preds.columns) - 1}")
print(f"  Shape: {target_preds.shape}")

print(f"\n  Target IDs (Ensembl, first 10):")
for i, target in enumerate(target_preds['targetId'].head(10)):
    print(f"    {i+1}. {target}")

# ============================================================================
# Part 4: Run Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING DRUG-TARGET MAPPING EVALUATION")
print("=" * 80)
print("\nEvaluating using LINCS mode (default)")
print("This filters to drug-target pairs present in high-quality LINCS data\n")

try:
    results = submit_drug_target_mapping(drug_preds, target_preds, mode='lincs')
    
    # ============================================================================
    # Part 5: Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    print("\nData Sources:")
    print(f"  Drugs (trt_cp):   {len(drug_preds):,} compounds with {len(drug_preds.columns)-1} gene features")
    print(f"  Targets (trt_sh): {len(target_preds):,} genes with {len(target_preds.columns)-1} gene features")
    
    print("\nKey Metrics:")
    print(f"  AUROC:                     {results.get('auroc', 0):.4f}")
    print(f"  AUPRC:                     {results.get('auprc', 0):.4f}")
    print(f"  Drug->Target Precision@10: {results.get('drug_to_target_precision@10', 0):.4f}")
    print(f"  Target->Drug Precision@10: {results.get('target_to_drug_precision@10', 0):.4f}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("\nThis evaluation uses real LINCS L1000 gene expression profiles to assess")
    print("whether expression-based representations can predict known drug-target interactions.")

except Exception as e:
    print("\n" + "!" * 80)
    print("! EVALUATION FAILED")
    print("!" * 80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
