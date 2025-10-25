import os
import pandas as pd
import numpy as np
import torch
import anndata as ad
from tqdm import tqdm
import sys
import gc

from contextpert import submit_drug_disease_cohesion
from contextpert.utils import canonicalize_smiles

# Add AIDO.Cell path - must be done before importing cell_utils
sys.path.insert(0, 'data_download/ModelGenerator/experiments/AIDO.Cell')
import cell_utils
from modelgenerator.tasks import Embed

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


print("="*80)
print("AIDO.CELL EMBEDDING EVALUATION")
print("="*80)
print("\nThis example uses AIDO.Cell embeddings generated from LINCS L1000 expression")
print("profiles as molecular representations for the evaluation framework.\n")

# OPTIMIZATION: Load reference SMILES first to filter LINCS data early
print("Loading reference SMILES for LINCS evaluation...")
ref_df = pd.read_csv(os.path.join(DATA_DIR, 'opentargets/disease_drug_triples_csv/disease_drug_triples_lincs.csv'))
print(f"  Reference set contains {len(ref_df['smiles'].unique())} unique SMILES for evaluation")

# Canonicalize reference SMILES for matching
print("  Canonicalizing reference SMILES...")
ref_df['smiles_canon'] = ref_df['smiles'].apply(canonicalize_smiles)
ref_smiles_canon = set(ref_df['smiles_canon'].unique())
print(f"  {len(ref_smiles_canon)} unique canonical SMILES\n")

# Load LINCS data and filter to evaluation set
print(f"Loading LINCS data from: {DATA_DIR}/trt_cp_smiles.csv")
print("  This will take a moment...")

lincs_df_full = pd.read_csv(os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv'))
print(f"  Loaded {len(lincs_df_full):,} total perturbation profiles")

# Filter out invalid SMILES
print("  Filtering to evaluation set...")
bad_smiles = ['-666', 'restricted']
lincs_df_full = lincs_df_full[~lincs_df_full['canonical_smiles'].isin(bad_smiles)]
lincs_df_full = lincs_df_full[lincs_df_full['canonical_smiles'].notna()]

# Canonicalize and filter to reference set
lincs_df_full['smiles_canon'] = lincs_df_full['canonical_smiles'].apply(canonicalize_smiles)
lincs_df = lincs_df_full[lincs_df_full['smiles_canon'].isin(ref_smiles_canon)].copy()

# Delete the full dataframe immediately
del lincs_df_full
gc.collect()

print(f"  After filtering to evaluation set: {len(lincs_df):,} samples")
print(f"  Unique BRD compounds: {lincs_df['pert_id'].nunique():,}")
print(f"  Unique SMILES: {lincs_df['canonical_smiles'].nunique():,}\n")

if len(lincs_df) == 0:
    raise ValueError("No overlap between LINCS and reference SMILES!")

# Initialize AIDO.Cell model
print("Initializing AIDO.Cell model...")
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
backbone = 'aido_cell_3m'

print(f"  Loading {backbone} model on {device}...")
model = Embed.from_config({
    "model.backbone": backbone,
    "model.batch_size": batch_size
}).eval()
model = model.to(device).to(torch.bfloat16)
print(f"  Model loaded successfully!\n")

# Identify gene expression columns (numeric columns that are Entrez IDs)
metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                 'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                 'pct_self_rank_q25', 'canonical_smiles', 'inchi_key']
gene_cols = [col for col in lincs_df.columns if col not in metadata_cols]

print(f"  Gene expression features: {len(gene_cols)} (landmark genes as Entrez IDs)")

# Convert Entrez IDs to gene symbols for AIDO.Cell
print("\nConverting Entrez IDs to gene symbols...")
entrez_map = pd.read_csv(os.path.join(DATA_DIR, 'entrez_to_ensembl_map.csv'))
# Note: gene_cols are strings but entrez_id is int, so convert to int for matching
entrez_to_symbol = dict(zip(entrez_map['entrez_id'], entrez_map['symbol']))

# Create mapping for column renaming - convert string column names to int for lookup
col_mapping = {}
unmapped = []
for col in gene_cols:
    try:
        entrez_int = int(col)
        if entrez_int in entrez_to_symbol:
            col_mapping[col] = entrez_to_symbol[entrez_int]
        else:
            unmapped.append(col)
    except ValueError:
        unmapped.append(col)

if unmapped:
    print(f"  Warning: {len(unmapped)} genes not found in mapping (out of {len(gene_cols)})")
print(f"  Successfully mapped {len(col_mapping)}/{len(gene_cols)} genes to symbols")

# Prepare expression data with gene symbols - keep only necessary columns
print("\nPreparing expression matrix...")
symbol_cols = [col_mapping[c] for c in gene_cols if c in col_mapping]

# Extract data directly to avoid extra copies - use canonicalized SMILES
cell_ids = lincs_df['inst_id'].values
pert_ids = lincs_df['pert_id'].values
canonical_smiles = lincs_df['smiles_canon'].values

# Get expression matrix with mapped columns
expr_cols = [c for c in gene_cols if c in col_mapping]
expr_matrix = lincs_df[expr_cols].values

print(f"  Expression matrix shape: {expr_matrix.shape}")

# Free lincs_df to save memory
del lincs_df
gc.collect()

print("\nGenerating AIDO.Cell embeddings...")
print("  Model: aido_cell_3m (128 dimensions)")
print("  This may take several minutes...")

# Create AnnData object
print("  Creating AnnData object...")
adata = ad.AnnData(X=expr_matrix)
adata.obs['cell_id'] = cell_ids
adata.var_names = symbol_cols
adata.obs_names = cell_ids

# Free expr_matrix now that it's in adata
del expr_matrix
gc.collect()

# Align to AIDO.Cell input format
print("  Aligning genes to AIDO.Cell vocabulary...")
aligned_adata, attention_mask = cell_utils.align_adata(adata)

# Free original adata
del adata
gc.collect()

# Prepare data (model already loaded above)
X = aligned_adata.X.astype(np.float32)
n_samples = X.shape[0]

# Pre-allocate array for mean embeddings to save memory
# We'll compute mean on-the-fly instead of storing full embeddings
embedding_dim = 128  # aido_cell_3m has 128 dimensions
mean_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)

# Generate embeddings in batches
print(f"  Processing {n_samples} samples in batches of {batch_size}...")
for i in tqdm(range(0, n_samples, batch_size)):
    batch_np = X[i:i+batch_size]
    batch_tensor = torch.from_numpy(batch_np).to(torch.bfloat16).to(device)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long).to(device)
    attention_mask_tensor = attention_mask_tensor.unsqueeze(0).expand(batch_tensor.size(0), -1)

    with torch.no_grad():
        transformed = model.transform({'sequences': batch_tensor, 'attention_mask': attention_mask_tensor})
        embs = model(transformed)  # (batch_size, sequence_length, hidden_dim)
        # Compute mean across sequence dimension immediately and move to CPU
        batch_mean = embs.mean(dim=1).to(dtype=torch.float32).cpu().numpy()
        mean_embeddings[i:i+len(batch_mean)] = batch_mean

        # Clear GPU memory
        del batch_tensor, attention_mask_tensor, transformed, embs, batch_mean
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"  Generated embeddings shape: {mean_embeddings.shape}")

# Create dataframe with inst_id, pert_id, canonical_smiles, and embeddings
# Build all columns at once to avoid fragmentation
print("  Building embedding dataframe...")
emb_dict = {
    'inst_id': cell_ids,
    'pert_id': pert_ids,
    'canonical_smiles': canonical_smiles
}
# Add all embedding dimensions at once
for i in range(mean_embeddings.shape[1]):
    emb_dict[f'emb_{i}'] = mean_embeddings[:, i]

embedding_df = pd.DataFrame(emb_dict)
print(f"  Embedding dataframe shape: {embedding_df.shape}")

# Aggregate embeddings by BRD ID (average across all perturbations of same compound)
print("\nAggregating embeddings by BRD ID...")
emb_cols = [f'emb_{i}' for i in range(embedding_dim)]

# Free embeddings array
del mean_embeddings
gc.collect()
emb_by_brd = (
    embedding_df.groupby('pert_id')[emb_cols + ['canonical_smiles']]
    .agg({**{col: 'mean' for col in emb_cols}, 'canonical_smiles': 'first'})
    .reset_index()
)

print(f"Created embedding representations for {len(emb_by_brd)} unique BRD compounds")

# SMILES are already canonical, just rename the column
emb_by_brd = emb_by_brd.rename(columns={'canonical_smiles': 'smiles'})
print(f"  Final compounds: {len(emb_by_brd):,}")

# Prepare prediction dataframe: SMILES + embedding features
pred_data = {'smiles': emb_by_brd['smiles'].values}
for emb_col in emb_cols:
    pred_data[emb_col] = emb_by_brd[emb_col].values

my_preds = pd.DataFrame(pred_data)

print(f"\nFinal prediction dataframe:")
print(f"  Unique compounds: {len(my_preds)}")
print(f"  Representation dimensionality: {len(emb_cols)} dimensions")
print(f"  Shape: {my_preds.shape}")
print(f"\nFirst few rows:")
print(my_preds.iloc[:3, :5])  # Show first 3 rows, first 5 columns

# Submit for evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION (LINCS MODE)")
print("="*80)
print("Using 'lincs' mode to evaluate only on drugs present in both LINCS and OpenTargets")

results = submit_drug_disease_cohesion(my_preds, mode='lincs')

print("\nEvaluation complete! These are results using AIDO.Cell embeddings from LINCS L1000.")
