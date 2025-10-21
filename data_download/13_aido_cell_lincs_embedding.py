import os
import pandas as pd
import numpy as np
import torch
import anndata as ad
from tqdm import tqdm
import mygene
import sys
import gc
from typing import List, Set, Dict

# Add AIDO.Cell path - must be done before importing cell_utils
sys.path.insert(0, 'ModelGenerator/experiments/AIDO.Cell')
import cell_utils
from modelgenerator.tasks import Embed

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


lincs_path = os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv')
# lincs_path = os.path.join(DATA_DIR, 'lincs_small.csv')
print(f"\nLoading compound perturbation data from: {lincs_path}")
lincs_df = pd.read_csv(lincs_path)

# Identify gene expression columns (Entrez IDs)
metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                 'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                 'pct_self_rank_q25', 'canonical_smiles']
                #  'pct_self_rank_q25']
gene_cols = [col for col in lincs_df.columns if col not in metadata_cols]
metadata_df = lincs_df[metadata_cols]

print(f"  Gene expression features: {len(gene_cols)} (landmark genes as Entrez IDs)")

# Convert Entrez IDs to gene symbols for AIDO.Cell
# Identify columns that are Entrez IDs (represented as numeric strings)
# Query mygene.info API
mg = mygene.MyGeneInfo()
gene_info = mg.querymany(
    gene_cols,
    scopes="entrezgene",
    fields="symbol",
    species="human",
    as_dataframe=False,
)

# Create a mapping from Entrez ID to symbol
id_to_symbol_map: Dict[str, str] = {
    str(item["query"]): item.get("symbol", item["query"])
    for item in gene_info
    if "symbol" in item
}

# Drop unmapped genes from gene_cols
mapped_gene_cols = [col for col in gene_cols if col in id_to_symbol_map]
unmapped = set(gene_cols) - set(mapped_gene_cols)
if unmapped:
    print(f"  Warning: {len(unmapped)} genes not found in mapping (out of {len(gene_cols)})")
print(f"  Successfully mapped {len(mapped_gene_cols)}/{len(gene_cols)} genes to symbols")

# Rename DataFrame columns 
lincs_df.rename(columns=id_to_symbol_map, inplace=True)

# Prepare expression data with gene symbols - keep only necessary columns
print("\nPreparing expression matrix...")
symbol_cols = [id_to_symbol_map[col] for col in mapped_gene_cols]

# Extract data directly to avoid extra copies - use canonicalized SMILES
cell_ids = lincs_df['inst_id'].values
pert_ids = lincs_df['pert_id'].values

# Get expression matrix with mapped columns
expr_matrix = lincs_df[symbol_cols].values
print(f"  Expression matrix shape: {expr_matrix.shape}")

# Free lincs_df to save memory
del lincs_df
gc.collect()

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

# Pre-allocate array for mean embeddings to save memory
# We'll compute mean on-the-fly instead of storing full embeddings
if backbone == 'aido_cell_3m':
    embedding_dim = 128
elif backbone == 'aido_cell_10m':
    embedding_dim = 256
elif backbone == 'aido_cell_100m':
    embedding_dim = 650
mean_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)

print("\nGenerating AIDO.Cell embeddings...")
print(f"  Model: {backbone} ({embedding_dim} dimensions)")
print("  This may take several minutes...")

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
print(f"  Generated embeddings shape: {mean_embeddings.shape}")

# Clean up unused vars before saving
del model
del aligned_adata, attention_mask
gc.collect()

# Create embedding df, merge with metadata_df
embedding_df = pd.DataFrame(mean_embeddings, columns=[f'emb_{j}' for j in range(embedding_dim)])
result_df = pd.concat([metadata_df.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)

# Save to CSV
output_path = os.path.join(DATA_DIR, f'lincs_{backbone}_embeddings.csv')
print(f"\nSaving embeddings with metadata to: {output_path}")
result_df.to_csv(output_path, index=False)