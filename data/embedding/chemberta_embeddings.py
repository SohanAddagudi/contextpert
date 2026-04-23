"""
Generate ChemBERTa embeddings for all inst_ids in trt_cp_smiles_qc.csv.
Outputs: data/gene_embeddings/chemberta_embeddings.npz
  - inst_ids: array of inst_id strings
  - embeddings: array of shape (N, embedding_dim)
"""
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])
OUTPUT_FILE = DATA_DIR / "gene_embeddings" / "chemberta_embeddings.npz"

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
BATCH_SIZE = 256

print(f"Loading data from {DATA_DIR / 'trt_cp_smiles_qc.csv'}...")
df = pd.read_csv(DATA_DIR / "trt_cp_smiles_qc.csv")
df = df[["inst_id", "canonical_smiles"]].dropna(subset=["canonical_smiles"])
df = df[df["canonical_smiles"] != ""]
print(f"  {len(df)} rows with valid SMILES, {df['canonical_smiles'].nunique()} unique SMILES")

print(f"Loading ChemBERTa model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
print(f"  Using device: {device}")

unique_smiles = df["canonical_smiles"].unique()
print(f"Computing embeddings for {len(unique_smiles)} unique SMILES...")

smiles_to_emb = {}
for i in range(0, len(unique_smiles), BATCH_SIZE):
    batch = unique_smiles[i : i + BATCH_SIZE].tolist()
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # mean-pool over token dimension
    embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    for smi, emb in zip(batch, embs):
        smiles_to_emb[smi] = emb
    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  {i + len(batch)}/{len(unique_smiles)} done")

print("Mapping embeddings back to inst_ids...")
inst_ids = df["inst_id"].values
embeddings = np.array([smiles_to_emb[smi] for smi in df["canonical_smiles"].values], dtype=np.float32)

print(f"Saving to {OUTPUT_FILE}  shape={embeddings.shape}")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUTPUT_FILE, inst_ids=inst_ids, embeddings=embeddings)
print("Done.")
