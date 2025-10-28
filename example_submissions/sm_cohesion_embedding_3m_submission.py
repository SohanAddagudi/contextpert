import os
import pandas as pd

from contextpert import submit_drug_disease_cohesion
from contextpert.utils import canonicalize_smiles

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


print("="*80)
print("AIDO CELL 3M EMBEDDING EVALUATION")
print("="*80)
print("\nThis example uses AIDO Cell 3M embeddings (128-dim) from LINCS L1000 data")
print("as molecular representations for the evaluation framework.\n")

# Load LINCS L1000 compound perturbation embedding data with canonical SMILES
print(f"Loading LINCS compound perturbation embeddings from: {DATA_DIR}/trt_cp_smiles_qc_aido_cell_3m_embeddings.csv")
lincs_df = pd.read_csv(os.path.join(DATA_DIR, 'trt_cp_smiles_qc_aido_cell_3m_embeddings.csv'))

print(f"Loaded {len(lincs_df):,} perturbation profiles")
print(f"  Unique BRD compounds: {lincs_df['pert_id'].nunique():,}")

# Filter out invalid SMILES
print("\nFiltering valid SMILES...")
bad_smiles = ['-666', 'restricted']
lincs_df = lincs_df[~lincs_df['canonical_smiles'].isin(bad_smiles)].copy()
lincs_df = lincs_df[lincs_df['canonical_smiles'].notna()].copy()

print(f"  Remaining samples with valid SMILES: {len(lincs_df):,}")
print(f"  Unique BRD compounds: {lincs_df['pert_id'].nunique():,}")

# Identify embedding columns
metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                 'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                 'pct_self_rank_q25', 'canonical_smiles', 'inchi_key']
embedding_cols = [col for col in lincs_df.columns if col.startswith('emb_')]

print(f"  Embedding features: {len(embedding_cols)} dimensions")

# Aggregate embeddings by BRD ID (average across all perturbations of same compound)
print("\nAggregating embeddings by BRD ID...")
embed_by_brd = (
    lincs_df.groupby('pert_id')[embedding_cols + ['canonical_smiles']]
    .agg({**{col: 'mean' for col in embedding_cols}, 'canonical_smiles': 'first'})
    .reset_index()
)

print(f"Created embedding representations for {len(embed_by_brd)} unique BRD compounds")

# Canonicalize SMILES
print("\nCanonicalizing SMILES for consistent comparison...")
failed_canon = []

def safe_canonicalize(smiles):
    try:
        return canonicalize_smiles(smiles)
    except:
        failed_canon.append(smiles)
        return None

embed_by_brd['smiles'] = embed_by_brd['canonical_smiles'].apply(safe_canonicalize)
embed_by_brd = embed_by_brd[embed_by_brd['smiles'].notna()].copy()

if failed_canon:
    print(f"  Warning: {len(failed_canon)} SMILES failed canonicalization")

print(f"  Final compounds with canonical SMILES: {len(embed_by_brd):,}")

# Prepare prediction dataframe: SMILES + embedding features
pred_data = {'smiles': embed_by_brd['smiles'].values}
for embed_col in embedding_cols:
    pred_data[embed_col] = embed_by_brd[embed_col].values

my_preds = pd.DataFrame(pred_data)

print(f"\nFinal prediction dataframe:")
print(f"  Unique compounds: {len(my_preds)}")
print(f"  Representation dimensionality: {len(embedding_cols)} embedding dimensions")
print(f"  Shape: {my_preds.shape}")
print(f"\nFirst few rows:")
print(my_preds.iloc[:3, :5])  # Show first 3 rows, first 5 columns

# Submit for evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION (LINCS MODE)")
print("="*80)
print("Using 'lincs' mode to evaluate only on drugs present in both LINCS and OpenTargets")

results = submit_drug_disease_cohesion(my_preds, mode='lincs')

print("\nEvaluation complete! These are results using AIDO Cell 3M embeddings (128-dim).")
