import os
import pandas as pd
import numpy as np  # Added for loading model outputs

from contextpert import submit_drug_disease_cohesion
from contextpert.utils import canonicalize_smiles

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']
LINCS_META_PATH = os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv')

# Construct paths to the specific model's output files
MODEL_RESULTS_DIR = os.path.join(DATA_DIR, 'sm_cohesion_network/trt_cp_AIDOcell/bs_64_arch_10/')
BETA_PATH = os.path.join(MODEL_RESULTS_DIR, 'full_dataset_betas.npy')
PRED_CSV_PATH = os.path.join(MODEL_RESULTS_DIR, 'full_dataset_predictions.csv')


print("="*80)
print(f"MODEL PARAMETER EVALUATION")
print("="*80)
print("\nThis example uses learned model parameters (betas) from a")
print(f"ContextualizedCorrelation model as molecular representations.\n")

print(f"Loading model betas from: {BETA_PATH}")
betas = np.load(BETA_PATH)

n_samples, n_pcs, _ = betas.shape
n_features = n_pcs * n_pcs
print(f"  Loaded betas for {n_samples:,} samples")
print(f"  Original shape: ({n_samples}, {n_pcs}, {n_pcs})")

# Flatten betas into feature vectors
print(f"Flattening betas into feature vectors of length {n_features:,}...")
betas_flat = betas.reshape(n_samples, n_features)
feature_cols = [f'beta_{i}' for i in range(n_features)]
beta_df = pd.DataFrame(betas_flat, columns=feature_cols)

# Load model prediction metadata (for inst_id)
print(f"Loading prediction metadata from: {PRED_CSV_PATH}")
pred_meta_df = pd.read_csv(PRED_CSV_PATH)

if len(pred_meta_df) != len(beta_df):
    print(f"Error: Mismatch in length between betas ({len(beta_df)}) and metadata ({len(pred_meta_df)})")
    exit()


model_data_df = pd.concat([pred_meta_df[['inst_id']], beta_df], axis=1)
print(f"Combined model data shape: {model_data_df.shape}")


print(f"Loading LINCS metadata (for SMILES) from: {LINCS_META_PATH}")
try:
    lincs_meta_df = pd.read_csv(
        LINCS_META_PATH,
        usecols=['inst_id', 'pert_id', 'canonical_smiles']
    )
except FileNotFoundError:
    print(f"Error: LINCS metadata file not found at {LINCS_META_PATH}")
    print("Please check your 'DATA_DIR' and 'LINCS_META_PATH' variables.")
    exit()

print(f"Loaded metadata for {len(lincs_meta_df):,} instances")

# Merge model data with LINCS metadata on 'inst_id'
print("\nMerging model data with SMILES metadata on 'inst_id'...")
merged_df = pd.merge(model_data_df, lincs_meta_df, on='inst_id', how='left')

print(f"  Samples before merge: {len(model_data_df):,}")
print(f"  Samples after merge:  {len(merged_df):,}")
print(f"  (Note: Samples from the model run not found in '{os.path.basename(LINCS_META_PATH)}' will be dropped)")


# Filter out invalid SMILES
print("\nFiltering valid SMILES...")
bad_smiles = ['-666', 'restricted']
merged_df = merged_df[~merged_df['canonical_smiles'].isin(bad_smiles)].copy()
merged_df = merged_df[merged_df['canonical_smiles'].notna()].copy()

print(f"  Remaining samples with valid SMILES: {len(merged_df):,}")
print(f"  Unique BRD compounds: {merged_df['pert_id'].nunique():,}")

# Aggregate model representations by BRD ID (average across all perturbations of same compound)
print("\nAggregating model (beta) representations by BRD ID...")
agg_cols = feature_cols + ['canonical_smiles']

model_rep_by_brd = (
    merged_df.groupby('pert_id')[agg_cols]
    .agg({**{col: 'mean' for col in feature_cols}, 'canonical_smiles': 'first'})
    .reset_index()
)

print(f"Created model representations for {len(model_rep_by_brd)} unique BRD compounds")

# Canonicalize SMILES
print("\nCanonicalizing SMILES for consistent comparison...")
failed_canon = []

def safe_canonicalize(smiles):
    try:
        return canonicalize_smiles(smiles)
    except:
        failed_canon.append(smiles)
        return None

model_rep_by_brd['smiles'] = model_rep_by_brd['canonical_smiles'].apply(safe_canonicalize)
model_rep_by_brd = model_rep_by_brd[model_rep_by_brd['smiles'].notna()].copy()

if failed_canon:
    print(f"  Warning: {len(failed_canon)} SMILES failed canonicalization")

print(f"  Final compounds with canonical SMILES: {len(model_rep_by_brd):,}")


pred_data = {'smiles': model_rep_by_brd['smiles'].values}
for col in feature_cols:
    pred_data[col] = model_rep_by_brd[col].values

my_preds = pd.DataFrame(pred_data)

print(f"\nFinal prediction dataframe:")
print(f"  Unique compounds: {len(my_preds)}")
print(f"  Representation dimensionality: {len(feature_cols)} features (from {n_pcs}x{n_pcs} betas)")
print(f"  Shape: {my_preds.shape}")
print(f"\nFirst few rows:")
print(my_preds.iloc[:3, :5])  # Show first 3 rows, first 5 columns

# Submit for evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION (LINCS MODE)")
print("="*80)
print("Using 'lincs' mode to evaluate only on drugs present in both LINCS and OpenTargets")

results = submit_drug_disease_cohesion(my_preds, mode='lincs')

print(f"\nEvaluation complete! These are results using model parameters (betas) from '{EMB_NAME}'.")
