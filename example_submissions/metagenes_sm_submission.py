import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from contextpert import submit_drug_disease_cohesion
from contextpert.utils import canonicalize_smiles

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


print("="*80)
print("METAGENE EVALUATION")
print("="*80)
print("\nThis example uses PCA-compressed gene expression profiles (metagenes)")
print("from LINCS L1000 data as molecular representations.\n")

# Load LINCS L1000 compound perturbation data with canonical SMILES
print(f"Loading LINCS compound perturbation data from: {DATA_DIR}/trt_cp_smiles.csv")
lincs_df = pd.read_csv(os.path.join(DATA_DIR, 'trt_cp_smiles.csv'))

print(f"Loaded {len(lincs_df):,} perturbation profiles")
print(f"  Unique BRD compounds: {lincs_df['pert_id'].nunique():,}")

# Filter out invalid SMILES
print("\nFiltering valid SMILES...")
bad_smiles = ['-666', 'restricted']
lincs_df = lincs_df[~lincs_df['canonical_smiles'].isin(bad_smiles)].copy()
lincs_df = lincs_df[lincs_df['canonical_smiles'].notna()].copy()

print(f"  Remaining samples with valid SMILES: {len(lincs_df):,}")
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
    lincs_df.groupby('pert_id')[gene_cols + ['canonical_smiles']]
    .agg({**{col: 'mean' for col in gene_cols}, 'canonical_smiles': 'first'})
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

# Add canonical SMILES to PCA data
print("\n" + "="*80)
print("ADDING SMILES TO METAGENE REPRESENTATIONS")
print("="*80)
expr_by_brd_pca = expr_by_brd_pca.merge(
    expr_by_brd[['pert_id', 'canonical_smiles']],
    on='pert_id',
    how='left'
)

# Canonicalize SMILES
print("Canonicalizing SMILES for consistent comparison...")
failed_canon = []

def safe_canonicalize(smiles):
    try:
        return canonicalize_smiles(smiles)
    except:
        failed_canon.append(smiles)
        return None

expr_by_brd_pca['smiles'] = expr_by_brd_pca['canonical_smiles'].apply(safe_canonicalize)
expr_by_brd_pca = expr_by_brd_pca[expr_by_brd_pca['smiles'].notna()].copy()

if failed_canon:
    print(f"  Warning: {len(failed_canon)} SMILES failed canonicalization")

print(f"  Final compounds with canonical SMILES: {len(expr_by_brd_pca):,}")

# Prepare prediction dataframe: SMILES + metagene features
metagene_cols = [f'metagene_{i}' for i in range(n_components)]
pred_data = {'smiles': expr_by_brd_pca['smiles'].values}
for col in metagene_cols:
    pred_data[col] = expr_by_brd_pca[col].values

my_preds = pd.DataFrame(pred_data)

print(f"\nFinal prediction dataframe:")
print(f"  Unique compounds: {len(my_preds)}")
print(f"  Representation dimensionality: {n_components} metagenes")
print(f"  Shape: {my_preds.shape}")
print(f"\nFirst few rows:")
print(my_preds.iloc[:3, :6])  # Show first 3 rows, first 6 columns

# Submit for evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION (LINCS MODE)")
print("="*80)
print("Using 'lincs' mode to evaluate only on drugs present in both LINCS and OpenTargets")

results = submit_drug_disease_cohesion(my_preds, mode='lincs')

print("\nEvaluation complete! These are results using PCA-compressed gene expression (metagenes) from LINCS L1000.")
print(f"Dimensionality reduction: {len(gene_cols)} genes → {n_components} metagenes")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
print("Evaluated on the intersection of LINCS and OpenTargets drugs.")
