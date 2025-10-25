import os
import pandas as pd

from contextpert import submit_drug_disease_cohesion
from contextpert.utils import canonicalize_smiles

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


print("="*80)
print("EXPRESSION EVALUATION")
print("="*80)
print("\nThis example uses gene expression profiles from LINCS L1000 data")
print("as molecular representations for the evaluation framework.\n")

# Load LINCS L1000 compound perturbation data with canonical SMILES
print(f"Loading LINCS compound perturbation data from: {DATA_DIR}/trt_cp_smiles.csv")
lincs_df = pd.read_csv(os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv'))

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

# Canonicalize SMILES
print("\nCanonicalizing SMILES for consistent comparison...")
failed_canon = []

def safe_canonicalize(smiles):
    try:
        return canonicalize_smiles(smiles)
    except:
        failed_canon.append(smiles)
        return None

expr_by_brd['smiles'] = expr_by_brd['canonical_smiles'].apply(safe_canonicalize)
expr_by_brd = expr_by_brd[expr_by_brd['smiles'].notna()].copy()

if failed_canon:
    print(f"  Warning: {len(failed_canon)} SMILES failed canonicalization")

print(f"  Final compounds with canonical SMILES: {len(expr_by_brd):,}")

# Prepare prediction dataframe: SMILES + expression features
pred_data = {'smiles': expr_by_brd['smiles'].values}
for gene_col in gene_cols:
    pred_data[f'gene_{gene_col}'] = expr_by_brd[gene_col].values

my_preds = pd.DataFrame(pred_data)

print(f"\nFinal prediction dataframe:")
print(f"  Unique compounds: {len(my_preds)}")
print(f"  Representation dimensionality: {len(gene_cols)} genes")
print(f"  Shape: {my_preds.shape}")
print(f"\nFirst few rows:")
print(my_preds.iloc[:3, :5])  # Show first 3 rows, first 5 columns

# Submit for evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION (LINCS MODE)")
print("="*80)
print("Using 'lincs' mode to evaluate only on drugs present in both LINCS and OpenTargets")

results = submit_drug_disease_cohesion(my_preds, mode='lincs')

print("\nEvaluation complete! These are results using gene expression profiles from LINCS L1000.")
