import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from contextpert import submit_sm_disease_cohesion
from contextpert.utils import brd_to_chembl_batch, chembl_to_smiles_batch

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


print("="*80)
print("METAGENE EVALUATION")
print("="*80)
print("\nThis example uses PCA-compressed gene expression profiles (metagenes)")
print("from LINCS L1000 data as molecular representations.\n")

# Load LINCS L1000 compound perturbation data
print(f"Loading LINCS compound perturbation data from: {DATA_DIR}/trt_cp_smiles.csv")
lincs_df = pd.read_csv(os.path.join(DATA_DIR, 'trt_cp_smiles.csv'))

print(f"Loaded {len(lincs_df):,} perturbation profiles")
print(f"  Unique BRD compounds: {lincs_df['pert_id'].nunique():,}")

# Identify gene expression columns (numeric columns that are Entrez IDs)
metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                 'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                 'pct_self_rank_q25', 'canonical_smiles', 'inchi_key']
gene_cols = [col for col in lincs_df.columns if col not in metadata_cols]

print(f"  Gene expression features: {len(gene_cols)} (landmark genes)")

# Aggregate expression profiles by BRD ID (average across all perturbations of same compound)
print("\nAggregating expression profiles by BRD ID...")
expr_by_brd = (
    lincs_df.groupby('pert_id')[gene_cols]
    .mean()
    .reset_index()
)

print(f"Created expression representations for {len(expr_by_brd)} unique BRD compounds")

# Apply PCA to compress to 50 metagenes
print("\n" + "="*80)
print("APPLYING PCA DIMENSIONALITY REDUCTION")
print("="*80)
n_components = 50
print(f"Compressing {len(gene_cols)} genes to {n_components} metagenes using PCA...")

# Standardize the data before PCA
scaler = StandardScaler()
expr_matrix = expr_by_brd[gene_cols].values
expr_scaled = scaler.fit_transform(expr_matrix)

# Apply PCA
pca = PCA(n_components=n_components, random_state=42)
expr_pca = pca.fit_transform(expr_scaled)

print(f"PCA complete!")
print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
print(f"  First 10 components explain: {pca.explained_variance_ratio_[:10].sum():.4f}")

# Create dataframe with PCA features
expr_by_brd_pca = expr_by_brd[['pert_id']].copy()
for i in range(n_components):
    expr_by_brd_pca[f'metagene_{i}'] = expr_pca[:, i]

# Map BRD IDs to ChEMBL IDs
print("\n" + "="*80)
print("MAPPING BRD IDs TO ChEMBL IDs")
print("="*80)
# Use existing cache from data_download/brd_chembl_cache if available
cache_dir = './data_download/brd_chembl_cache'
if not os.path.exists(cache_dir):
    cache_dir = None
    print("Note: No existing cache found, will query ChEMBL API (this may take a while)")
brd_to_chembl_map = brd_to_chembl_batch(
    expr_by_brd_pca['pert_id'],
    max_workers=20,
    cache_dir=cache_dir
)

# Merge mappings with PCA expression data
expr_with_chembl = expr_by_brd_pca.merge(
    brd_to_chembl_map[['brd_id', 'chembl_id']],
    left_on='pert_id',
    right_on='brd_id',
    how='left'
)

# Filter to compounds with valid ChEMBL IDs
expr_with_chembl = expr_with_chembl[expr_with_chembl['chembl_id'].notna()].copy()
print(f"\nMapped {len(expr_with_chembl)} compounds to ChEMBL IDs")

# Convert ChEMBL IDs to SMILES
print("\n" + "="*80)
print("CONVERTING ChEMBL IDs TO SMILES")
print("="*80)
chembl_ids = expr_with_chembl['chembl_id'].unique().tolist()
chembl_to_smiles = chembl_to_smiles_batch(chembl_ids, show_progress=True)

# Add SMILES to dataframe
expr_with_chembl['smiles'] = expr_with_chembl['chembl_id'].map(chembl_to_smiles)

# Filter to compounds with valid SMILES
expr_with_chembl = expr_with_chembl[expr_with_chembl['smiles'].notna()].copy()
print(f"\nObtained SMILES for {len(expr_with_chembl)} compounds")

# Prepare prediction dataframe: SMILES + metagene features
metagene_cols = [f'metagene_{i}' for i in range(n_components)]
pred_data = {'smiles': expr_with_chembl['smiles'].values}
for col in metagene_cols:
    pred_data[col] = expr_with_chembl[col].values

my_preds = pd.DataFrame(pred_data)

print(f"\nFinal prediction dataframe:")
print(f"  Unique compounds: {len(my_preds)}")
print(f"  Representation dimensionality: {n_components} metagenes")
print(f"  Shape: {my_preds.shape}")
print(f"\nFirst few rows:")
print(my_preds.iloc[:3, :6])  # Show first 3 rows, first 6 columns

# Submit for evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION")
print("="*80)

results = submit_sm_disease_cohesion(my_preds)

print("\nEvaluation complete! These are results using PCA-compressed gene expression (metagenes) from LINCS L1000.")
print(f"Dimensionality reduction: {len(gene_cols)} genes → {n_components} metagenes")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
