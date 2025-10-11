import os
import pandas as pd

from contextpert import submit_sm_disease_cohesion
from contextpert.utils import brd_to_chembl_batch, chembl_to_smiles_batch

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


print("="*80)
print("EXPRESSION EVALUATION")
print("="*80)
print("\nThis example uses gene expression profiles from LINCS L1000 data")
print("as molecular representations for the evaluation framework.\n")

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
    expr_by_brd['pert_id'],
    max_workers=20,
    cache_dir=cache_dir
)

# Merge mappings with expression data
expr_with_chembl = expr_by_brd.merge(
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

# Prepare prediction dataframe: SMILES + expression features
pred_data = {'smiles': expr_with_chembl['smiles'].values}
for gene_col in gene_cols:
    pred_data[f'gene_{gene_col}'] = expr_with_chembl[gene_col].values

my_preds = pd.DataFrame(pred_data)

print(f"\nFinal prediction dataframe:")
print(f"  Unique compounds: {len(my_preds)}")
print(f"  Representation dimensionality: {len(gene_cols)} genes")
print(f"  Shape: {my_preds.shape}")
print(f"\nFirst few rows:")
print(my_preds.iloc[:3, :5])  # Show first 3 rows, first 5 columns

# Submit for evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION")
print("="*80)

results = submit_sm_disease_cohesion(my_preds)

print("\nEvaluation complete! These are results using gene expression profiles from LINCS L1000.")
