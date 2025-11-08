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

from contextpert import submit_drug_target_mapping
from contextpert.utils import canonicalize_smiles

# --- Configuration ---
DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']
#dir for trt_cp networks
TRAINING_OUTPUT_PARENT_DIR = '/home/user/screening2/contextpert/all_run_results'

#dir for trt_sh networks
TRAINING_OUTPUT_PARENT_SH = '/home/user/screening2/contextpert/prev_inferences'
# ---------------------

print("=" * 80)
print("DRUG-TARGET BETA EVALUATION")
print("=" * 80)
print("\nThis example uses learned model parameters (betas) to evaluate")
print("drug-target interaction prediction:")
print("  - Drugs: Betas from compound perturbation models (trt_cp)")
print("  - Targets: Betas from shRNA knockdown models (trt_sh)")
print()


print("=" * 80)
print("LOADING DRUG DATA (COMPOUND PERTURBATION BETAS)")
print("=" * 80)
pd.set_option('display.max_columns', None)
# --- Define model paths ---
CP_PERT_NAME = 'trt_cp'
CP_EMB_NAME = 'AIDOprot'
CP_PARAM_NAME = 'bs_128_arch_50'
CP_MODEL_RESULTS_DIR = os.path.join(TRAINING_OUTPUT_PARENT_DIR, f'{CP_PERT_NAME}_{CP_EMB_NAME}', CP_PARAM_NAME)
CP_BETA_PATH = os.path.join(CP_MODEL_RESULTS_DIR, 'full_dataset_betas.npy')
CP_PRED_CSV_PATH = os.path.join(CP_MODEL_RESULTS_DIR, 'full_dataset_predictions.csv')

# --- Define model paths ---
SH_MODEL_RESULTS_DIR = os.path.join(TRAINING_OUTPUT_PARENT_SH, 'trt_sh_AIDOprot')
SH_BETA_PATH = os.path.join(SH_MODEL_RESULTS_DIR, 'full_dataset_betas.npy')
SH_PRED_CSV_PATH = os.path.join(SH_MODEL_RESULTS_DIR, 'full_dataset_predictions.csv')

# --- Load model betas and instance metadata ---
print(f"\nLoading model betas from: {CP_BETA_PATH}")
cp_betas = np.load(CP_BETA_PATH)
print(f"Loading prediction metadata from: {CP_PRED_CSV_PATH}")
cp_pred_meta_df = pd.read_csv(CP_PRED_CSV_PATH, usecols=['inst_id'])

# --- Flatten betas ---
n_samples, n_pcs, _ = cp_betas.shape
n_features = n_pcs * n_pcs
print(f"  Loaded betas for {n_samples:,} samples")
print(f"  Flattening betas into feature vectors of length {n_features:,}")
feature_cols = [f'beta_{i}' for i in range(n_features)]
cp_beta_df = pd.DataFrame(cp_betas.reshape(n_samples, n_features), columns=feature_cols)
cp_model_data_df = pd.concat([cp_pred_meta_df[['inst_id']], cp_beta_df], axis=1)

# --- Load LINCS metadata (using the same file as expression script) ---
trt_cp_path = os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv')
print(f"\nLoading LINCS metadata from: {trt_cp_path}")
lincs_meta_df = pd.read_csv(trt_cp_path)
print(lincs_meta_df)

# --- Merge betas with metadata ---
merged_df = pd.merge(cp_model_data_df, lincs_meta_df, on='inst_id', how='left')
bad_smiles = ['-666', 'restricted']
merged_df = merged_df[~merged_df['canonical_smiles'].isin(bad_smiles)]
merged_df = merged_df[merged_df['canonical_smiles'].notna()]
print(f"  Remaining samples with valid SMILES: {len(merged_df):,}")

# --- Aggregate by pert_id (same as expression script) ---
print("\nAggregating model (beta) representations by pert_id (BRD ID)")
agg_cols = feature_cols + ['canonical_smiles']
model_rep_by_brd = (
    merged_df.groupby('pert_id')[agg_cols]
    .agg({**{col: 'mean' for col in feature_cols}, 'canonical_smiles': 'first'})
    .reset_index()
)
print(f"  Aggregated to {len(model_rep_by_brd):,} unique compounds")

# --- Canonicalize SMILES ---
print("\nCanonicalizing SMILES...")
def safe_canonicalize(smiles):
    try:
        return canonicalize_smiles(smiles)
    except Exception:
        return None

model_rep_by_brd['smiles'] = model_rep_by_brd['canonical_smiles'].apply(safe_canonicalize)
model_rep_by_brd = model_rep_by_brd[model_rep_by_brd['smiles'].notna()].copy()

# --- Prepare final drug prediction dataframe ---
drug_preds = model_rep_by_brd[['smiles'] + feature_cols].copy()
print(f"\nFinal drug representation:")
print(f"  Unique compounds: {len(drug_preds)}")
print(f"  Features (betas): {len(feature_cols)}")
print(f"  Shape: {drug_preds.shape}")

# Clear memory
del cp_betas, cp_beta_df, cp_model_data_df, lincs_meta_df, merged_df, model_rep_by_brd

# ============================================================================
# Part 2: Load and Process Target Data (trt_sh Model Betas)
# ============================================================================
print("\n" + "=" * 80)
print("LOADING TARGET DATA (shRNA KNOCKDOWN BETAS)")
print("=" * 80)


# --- Load model betas and instance metadata ---
print(f"\nLoading model betas from: {SH_BETA_PATH}")
sh_betas = np.load(SH_BETA_PATH)
print(f"Loading prediction metadata from: {SH_PRED_CSV_PATH}")
sh_pred_meta_df = pd.read_csv(SH_PRED_CSV_PATH, usecols=['inst_id'])

# --- Flatten betas ---
sh_n_samples, sh_n_pcs, _ = sh_betas.shape
sh_n_features = sh_n_pcs * sh_n_pcs
print(f"  Loaded betas for {sh_n_samples:,} samples")
print(f"  Flattening betas into feature vectors of length {sh_n_features:,}...")
sh_feature_cols = [f'beta_{i}' for i in range(sh_n_features)]
sh_beta_df = pd.DataFrame(sh_betas.reshape(sh_n_samples, sh_n_features), columns=sh_feature_cols)
sh_model_data_df = pd.concat([sh_pred_meta_df[['inst_id']], sh_beta_df], axis=1)

# --- Load shRNA metadata (using the same annotated file as expression script) ---
trt_sh_genes_path = os.path.join(DATA_DIR, 'trt_sh_genes_qc.csv')
print(f"\nLoading shRNA metadata and target annotations from: {trt_sh_genes_path}")
# We only need inst_id to map to betas and ensembl_id for the target
sh_meta_df = pd.read_csv(trt_sh_genes_path, low_memory=False)
# print(sh_meta_df)

# --- Filter to only perturbations with target annotations ---
print(f"  Total instances in metadata: {len(sh_meta_df):,}")
sh_meta_df = sh_meta_df[sh_meta_df['ensembl_id'].notna()].copy()
print(f"  Retained instances with target annotations: {len(sh_meta_df):,}")
print(f"  Unique target genes: {sh_meta_df['ensembl_id'].nunique():,}")

# --- Merge betas with target annotations ---
print("\nMerging betas with target annotations...")
sh_merged_df = pd.merge(sh_model_data_df, sh_meta_df, on='inst_id', how='inner')
print(f"  Matched {len(sh_merged_df):,} beta samples to target genes")

# --- Aggregate by target gene (same as expression script) ---
print("\nAggregating betas by target gene (Ensembl ID)...")
target_final_df = (
    sh_merged_df.groupby('ensembl_id')[sh_feature_cols]
    .mean()
    .reset_index()
)
print(f"  Aggregated to {len(target_final_df):,} unique target genes")

# --- Prepare final target prediction dataframe ---
target_preds = target_final_df.rename(columns={'ensembl_id': 'targetId'})

print(f"\nFinal target representation:")
print(f"  Unique targets: {len(target_preds)}")
print(f"  Features (betas): {len(target_preds.columns) - 1}")
print(f"  Shape: {target_preds.shape}")

# Clear memory
del sh_betas, sh_beta_df, sh_model_data_df, sh_meta_df, sh_merged_df, target_final_df

# ============================================================================
# Part 3: Run Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING DRUG-TARGET MAPPING EVALUATION")
print("=" * 80)
print("\nEvaluating using LINCS mode (default)")

try:
    # Ensure feature counts match
    if len(drug_preds.columns) != len(target_preds.columns):
        raise ValueError(
            f"Feature mismatch! Drugs have {len(drug_preds.columns)-1} features, "
            f"Targets have {len(target_preds.columns)-1} features."
        )

    results = submit_drug_target_mapping(drug_preds, target_preds, mode='lincs')

    # ============================================================================
    # Part 4: Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print("\nData Sources:")
    print(f"  Drugs (trt_cp):   {len(drug_preds):,} compounds with {len(drug_preds.columns)-1} beta features")
    print(f"  Targets (trt_sh): {len(target_preds):,} genes with {len(target_preds.columns)-1} beta features")

    print("\nKey Metrics:")
    print(f"  AUROC:                     {results.get('auroc', 0):.4f}")
    print(f"  AUPRC:                     {results.get('auprc', 0):.4f}")
    print(f"  Drug->Target Precision@10: {results.get('drug_to_target_precision@10', 0):.4f}")
    print(f"  Target->Drug Precision@10: {results.get('target_to_drug_precision@10', 0):.4f}")


except Exception as e:
    print("\n" + "!" * 80)
    print("! EVALUATION FAILED")
    print("!" * 80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

