import os
import numpy as np
import pandas as pd
from pathlib import Path

from lightning import seed_everything

from contextpert import submit_drug_disease_cohesion
from contextpert.utils import canonicalize_smiles

SEED = 10
seed_everything(SEED, workers=True)

DATA_DIR  = Path(os.environ["CONTEXTPERT_DATA_DIR"])
MODEL_DIR = Path(os.environ.get("MODEL_DIR",
              str(DATA_DIR / "cellvs_molecule_networks" / "chemberta_model_outputs")))

print("=" * 70)
print(f"Model dir: {MODEL_DIR.resolve()}")
print("=" * 70)

BETAS_PATH = MODEL_DIR / "full_dataset_betas.npy"
MUS_PATH   = MODEL_DIR / "full_dataset_mus.npy"
PREDS_CSV  = MODEL_DIR / "full_dataset_predictions.csv"

print("\n=== Loading network predictions ===")
betas = np.load(BETAS_PATH)
mus   = np.load(MUS_PATH)
meta  = pd.read_csv(PREDS_CSV)
n_pred, n_x = len(meta), mus.shape[-1]
print(f"Loaded {n_pred:,} samples | betas: {betas.shape} | mus: {mus.shape}")

idx_upper = np.triu_indices(n_x, k=1)

bsq_raw = betas[:, idx_upper[0], idx_upper[1]] ** 2   
mus_raw = mus[:,   idx_upper[0], idx_upper[1]]        

print("\n=== Building drug representations ===")
smiles_col = meta["canonical_smiles"].apply(
    lambda s: canonicalize_smiles(s) if pd.notna(s) and s not in ("-666", "restricted") else None
)

def drug_mean(arr, prefix):
    cols = [f"{prefix}_{i}" for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=cols)
    df["smiles"] = smiles_col.values
    return df.dropna(subset=["smiles"]).groupby("smiles")[cols].mean().reset_index(), cols

drug_bsq, b_cols = drug_mean(bsq_raw, "b")
drug_mus, m_cols = drug_mean(mus_raw, "m")
merged = drug_bsq.merge(drug_mus, on="smiles")
print(f"Drug representations: {len(merged)} drugs × {len(b_cols) + len(m_cols)} features "
      f"({len(b_cols)} bsq + {len(m_cols)} mus)")


def std_norm(x):
    s = x.std(axis=0, keepdims=True)
    return x / np.where(s == 0, 1, s)

arr = np.hstack([std_norm(merged[b_cols].values),
                 std_norm(merged[m_cols].values)])
drug_rep = pd.DataFrame(arr, columns=[f"f_{i}" for i in range(arr.shape[1])])
drug_rep["smiles"] = merged["smiles"].values

print("\n=== Evaluating sm_cohesion ===")
results = submit_drug_disease_cohesion(drug_rep, mode="lincs")
