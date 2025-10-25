import os
import pandas as pd
import numpy as np
import torch
import anndata as ad
from tqdm import tqdm
import mygene
import sys
import gc
import argparse
from typing import List, Set, Dict

# Add AIDO.Cell path - must be done before importing cell_utils
sys.path.insert(0, 'ModelGenerator/experiments/AIDO.Cell')
import cell_utils
from modelgenerator.tasks import Embed

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate AIDO.Cell embeddings for LINCS data')
parser.add_argument('--backbone', type=str, required=True,
                    choices=['aido_cell_3m', 'aido_cell_10m', 'aido_cell_100m'],
                    help='AIDO.Cell backbone model to use')
parser.add_argument('--chunk-size', type=int, default=10000,
                    help='Number of rows to process per chunk (default: 10000)')
args = parser.parse_args()

input_file = 'trt_cp_smiles.csv'
lincs_path = os.path.join(DATA_DIR, input_file)
print(f"\nProcessing compound perturbation data from: {lincs_path}")
print(f"Chunk size: {args.chunk_size} rows")

# Initialize AIDO.Cell model first (before processing chunks)
print("\nInitializing AIDO.Cell model...")
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
backbone = args.backbone

# Determine embedding dimension
if backbone == 'aido_cell_3m':
    embedding_dim = 128
elif backbone == 'aido_cell_10m':
    embedding_dim = 256
elif backbone == 'aido_cell_100m':
    embedding_dim = 640

print(f"  Loading {backbone} model on {device}...")
model = Embed.from_config({
    "model.backbone": backbone,
    "model.batch_size": batch_size,
}).eval()
model = model.to(device).to(torch.bfloat16)
print(f"  Model loaded successfully!")
print(f"  Embedding dimension: {embedding_dim}\n")

# Get gene mapping (do this once upfront)
print("Setting up gene ID to symbol mapping...")
# Read just the header to get gene columns
header_df = pd.read_csv(lincs_path, nrows=0)
metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                 'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                 'pct_self_rank_q25', 'canonical_smiles']
gene_cols = [col for col in header_df.columns if col not in metadata_cols]
print(f"  Found {len(gene_cols)} gene expression features")

# Query mygene.info API for gene symbol mapping
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
print(f"  Successfully mapped {len(mapped_gene_cols)}/{len(gene_cols)} genes to symbols\n")

symbol_cols = [id_to_symbol_map[col] for col in mapped_gene_cols]

# Prepare output directory for chunks
input_filename_wo_ext = os.path.splitext(input_file)[0]
chunk_dir = os.path.join(DATA_DIR, f'{input_filename_wo_ext}_{backbone}_chunks')
os.makedirs(chunk_dir, exist_ok=True)
print(f"Chunk directory: {chunk_dir}\n")

# Process data in chunks
print("Processing data in chunks...")
chunk_files = []
chunk_num = 0

# Use chunksize parameter to read in chunks
for chunk_df in pd.read_csv(lincs_path, chunksize=args.chunk_size):
    chunk_num += 1
    print(f"\n{'='*60}")
    print(f"Processing chunk {chunk_num} ({len(chunk_df)} rows)...")
    print(f"{'='*60}")

    # Rename columns using the mapping
    chunk_df.rename(columns=id_to_symbol_map, inplace=True)

    # Extract metadata
    metadata_df = chunk_df[metadata_cols].copy()

    # Extract expression matrix
    expr_matrix = chunk_df[symbol_cols].values
    cell_ids = chunk_df['inst_id'].values

    # Free chunk_df
    del chunk_df
    gc.collect()

    # Create AnnData object
    print("  Creating AnnData object...")
    adata = ad.AnnData(X=expr_matrix)
    adata.obs['cell_id'] = cell_ids
    adata.var_names = symbol_cols
    adata.obs_names = cell_ids

    # Free expr_matrix
    del expr_matrix
    gc.collect()

    # Align to AIDO.Cell input format
    print("  Aligning genes to AIDO.Cell vocabulary...")
    aligned_adata, attention_mask = cell_utils.align_adata(adata)

    # Free original adata
    del adata
    gc.collect()

    # Prepare data for model
    X = aligned_adata.X.astype(np.float32)
    n_samples = X.shape[0]

    # Pre-allocate array for chunk embeddings
    chunk_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)

    # Generate embeddings
    print(f"  Generating embeddings for {n_samples} samples...")
    for i in tqdm(range(0, n_samples, batch_size), desc=f"  Chunk {chunk_num}"):
        batch_np = X[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch_np).to(torch.bfloat16).to(device)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long).to(device)
        attention_mask_tensor = attention_mask_tensor.unsqueeze(0).expand(batch_tensor.size(0), -1)

        with torch.no_grad():
            transformed = model.transform({'sequences': batch_tensor, 'attention_mask': attention_mask_tensor})
            embs = model(transformed)  # (batch_size, sequence_length, hidden_dim)
            # Compute mean across sequence dimension immediately and move to CPU
            batch_mean = embs.mean(dim=1).to(dtype=torch.float32).cpu().numpy()
            chunk_embeddings[i:i+len(batch_mean)] = batch_mean

    # Free memory
    del aligned_adata, attention_mask, X
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Create embedding dataframe and merge with metadata
    print("  Creating results dataframe...")
    embedding_df = pd.DataFrame(chunk_embeddings, columns=[f'emb_{j}' for j in range(embedding_dim)])
    result_df = pd.concat([metadata_df.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)

    # Save chunk to file
    chunk_file = os.path.join(chunk_dir, f'chunk_{chunk_num:04d}.csv')
    print(f"  Saving chunk to: {chunk_file}")
    result_df.to_csv(chunk_file, index=False)
    chunk_files.append(chunk_file)

    # Free memory
    del chunk_embeddings, embedding_df, result_df, metadata_df
    gc.collect()

    print(f"  Chunk {chunk_num} complete!")

# Clean up model to free memory before compilation
print("\n" + "="*60)
print("All chunks processed. Cleaning up model...")
print("="*60)
del model
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Compile all chunks into final output
print("\nCompiling chunks into final output...")
output_path = os.path.join(DATA_DIR, f'{input_filename_wo_ext}_{backbone}_embeddings.csv')
print(f"Final output: {output_path}")

# Read and concatenate all chunk files
print(f"Combining {len(chunk_files)} chunks...")
all_chunks = []
for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
    chunk = pd.read_csv(chunk_file)
    all_chunks.append(chunk)

final_df = pd.concat(all_chunks, ignore_index=True)
print(f"Final dataframe shape: {final_df.shape}")

# Save final output
print(f"Saving final output to: {output_path}")
final_df.to_csv(output_path, index=False)

# Clean up chunk files
print("\nCleaning up intermediate chunk files...")
for chunk_file in chunk_files:
    os.remove(chunk_file)
os.rmdir(chunk_dir)

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"Final embeddings saved to: {output_path}")
print(f"Total samples: {len(final_df)}")
