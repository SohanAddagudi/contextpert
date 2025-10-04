import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import torch
from modelgenerator.tasks import Embed
from tqdm import tqdm
from collections import defaultdict
import argparse
import os

import sys
sys.path.append('ModelGenerator/experiments/AIDO.Cell')
import cell_utils

parser = argparse.ArgumentParser(description='Generate embeddings using AIDO.Cell')
parser.add_argument('--input', required=True, help='Path to input CSV file (first column: ID, remaining columns: gene symbols)')
parser.add_argument('--output_base', required=True, help='Base directory for output files (will create {model}_lincs_embeddings.npy, {model}_lincs_gene_embeddings.npy, and aido_cell_lincs_gene_mask.npz)')
parser.add_argument('--model', required=True, choices=['aido_cell_3m', 'aido_cell_10m', 'aido_cell_100m'], help='Model to use for embeddings')
args = parser.parse_args()

# Load the CSV into a DataFrame
df = pd.read_csv(args.input)

# Assume the first column is cell_id
cell_ids = df.iloc[:, 0].values
expr_matrix = df.iloc[:, 1:].values
gene_names = df.columns[1:]

# Create AnnData object
adata = ad.AnnData(X=expr_matrix)

# Assign cell and gene metadata
adata.obs['cell_id'] = cell_ids
adata.var_names = gene_names
adata.obs_names = cell_ids

# Align to AIDO.Cell input format
aligned_adata, attention_mask = cell_utils.align_adata(adata)

# Get embeddings
batch_size=32
device='cuda'
backbone = args.model

# Initialize model
model = Embed.from_config({
    "model.backbone": backbone,
    "model.batch_size": batch_size
}).eval()
model = model.to(device).to(torch.bfloat16)

# Prepare data
X = aligned_adata.X.astype(np.float32)  # Convert to NumPy if sparse
n_samples = X.shape[0]
all_embeddings = []

# Loop in batches
for i in tqdm(range(0, n_samples, batch_size)):
    batch_np = X[i:i+batch_size]
    batch_tensor = torch.from_numpy(batch_np).to(torch.bfloat16).to(device)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long).to(device)
    attention_mask_tensor = attention_mask_tensor.unsqueeze(0).expand(batch_tensor.size(0), -1)

    with torch.no_grad():
        transformed = model.transform({'sequences': batch_tensor, 'attention_mask': attention_mask_tensor})
        embs = model(transformed)  # (batch_size, sequence_length, hidden_dim)
        pooled = embs.to(dtype=float).cpu().numpy()

    all_embeddings.append(pooled)

# Concatenate all batches
all_embeddings = np.vstack(all_embeddings)

# Generate output file names based on the output_base argument
output_base = args.output_base
os.makedirs(output_base, exist_ok=True)

# Save mean embeddings
mean_embeddings = all_embeddings.mean(axis=1)
np.save(os.path.join(output_base, f'{backbone}_lincs_embeddings.npy'), mean_embeddings)

# Save gene embeddings
cell_gene_embeddings = defaultdict(dict)
for i, cell_id in enumerate(cell_ids):
    for j, gene_name in enumerate(aligned_adata.var_names):
        cell_gene_embeddings[cell_id][gene_name] = all_embeddings[i, j]
np.save(os.path.join(output_base, f'{backbone}_lincs_gene_embeddings.npy'), cell_gene_embeddings)

# Save mask
valid = [gene_name for gene_name, mask in zip(aligned_adata.var_names, attention_mask) if mask]
masked = [gene_name for gene_name, mask in zip(aligned_adata.var_names, attention_mask) if not mask]
aido_gene_set = set(aligned_adata.var_names)
dropped = [gene_name for gene_name in gene_names if gene_name not in aido_gene_set]
np.savez(os.path.join(output_base, 'aido_cell_lincs_gene_mask.npz'), valid=valid, masked=masked, dropped=dropped)