"""
Train ridge regression predictors

For trt_cp (compound perturbations):
  Input:  Morgan fingerprints (2048-dim) computed from compound SMILES
  Targets: gene expression (977-dim), PCA metagenes (50-dim), AIDO Cell 3M embeddings (128-dim)
  Training: instances in train split of trt_cp_split_map.csv
  Prediction: all compounds with valid SMILES

For trt_sh (gene perturbations):
  Uses mean aggregation from train-split instances, with PCA fit on train genes only.
  Generates representations for all genes (train + test split).

Outputs saved to predictors/outputs/:
  cp_pred_expression.csv       - predicted expression (pert_id, smiles, gene_*)
  cp_pred_metagenes.csv        - predicted metagenes  (pert_id, smiles, metagene_*)
  cp_pred_aido_embeddings.csv  - predicted AIDO embs  (pert_id, smiles, emb_*)
  sh_repr_expression.csv       - aggregated sh expression   (targetId, gene_*)
  sh_repr_metagenes.csv        - aggregated sh metagenes    (targetId, metagene_*)
  sh_repr_aido_embeddings.csv  - aggregated sh AIDO embs    (targetId, emb_*)
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from contextpert.utils import canonicalize_smiles

DATA_DIR  = Path(os.environ["CONTEXTPERT_DATA_DIR"])
SPLIT_DIR = DATA_DIR / "gene_embeddings" / "unseen_perturbation_splits"
OUT_DIR   = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ALPHA    = 1.0   # Ridge regularization strength
N_PCA    = 50    # PCA components for metagenes
BAD_SMILES = {"-666", "restricted"}

META_COLS_CP = {"inst_id", "cell_id", "pert_id", "pert_type", "pert_dose",
                "pert_dose_unit", "pert_time", "sig_id", "distil_cc_q75",
                "pct_self_rank_q25", "canonical_smiles", "inchi_key"}
META_COLS_SH = {"inst_id", "cell_id", "pert_id", "pert_type", "pert_dose",
                "pert_dose_unit", "pert_time", "sig_id", "distil_cc_q75",
                "pct_self_rank_q25", "gene_symbol", "ensembl_id"}


# ── Morgan fingerprint utilities ───────────────────────────────────────────────

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def compute_morgan(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    arr = np.zeros(2048, dtype=np.float32)
    fp = _morgan_gen.GetFingerprint(mol)
    for i in range(2048):
        arr[i] = fp[i]
    return arr

def safe_canon(s):
    try:
        return canonicalize_smiles(s)
    except Exception:
        return None


# ==============================================================================
# Part 1: Compound perturbation (trt_cp) predictor
# ==============================================================================

print("=" * 70)
print("Part 1: Training compound perturbation predictor (trt_cp)")
print("=" * 70)

# Load data
print("Loading trt_cp_smiles_qc.csv...")
cp_df = pd.read_csv(DATA_DIR / "trt_cp_smiles_qc.csv")
gene_cols = [c for c in cp_df.columns if c not in META_COLS_CP]
cp_df = cp_df[~cp_df["canonical_smiles"].isin(BAD_SMILES)]
cp_df = cp_df[cp_df["canonical_smiles"].notna()].copy()
print(f"  {len(cp_df)} instances, {len(gene_cols)} gene columns")

# Load and merge splits
cp_split = pd.read_csv(SPLIT_DIR / "trt_cp_split_map.csv")
print(f"  Splits: {cp_split['split'].value_counts().to_dict()}")
cp_df = cp_df.merge(cp_split[["inst_id", "split"]], on="inst_id", how="left")

# Compute Morgan fingerprints for all drugs
print("Computing Morgan fingerprints...")
drug_info = cp_df.groupby("pert_id")["canonical_smiles"].first().reset_index()
drug_info["morgan"]  = drug_info["canonical_smiles"].apply(compute_morgan)
drug_info["smiles"]  = drug_info["canonical_smiles"].apply(safe_canon)
drug_info = drug_info.dropna(subset=["morgan", "smiles"]).reset_index(drop=True)
print(f"  {len(drug_info)} drugs with valid Morgan fingerprints")

# Aggregate train instances by drug
print("Aggregating train instances by drug...")
train_expr_by_drug = (
    cp_df[cp_df["split"] == "train"]
    .groupby("pert_id")[gene_cols]
    .mean()
    .reset_index()
)
print(f"  {len(train_expr_by_drug)} drugs in train split")

# Align Morgan FPs with train expression (inner join on pert_id)
train_aligned = drug_info[["pert_id", "smiles", "morgan"]].merge(
    train_expr_by_drug, on="pert_id", how="inner"
).reset_index(drop=True)
X_train = np.stack(train_aligned["morgan"].values)   # (n_train, 2048)
Y_expr  = train_aligned[gene_cols].values            # (n_train, 977)
print(f"  Training matrix: X={X_train.shape}, Y_expr={Y_expr.shape}")

# ---------- Expression predictor ----------
print("\n-- Fitting expression predictor --")
expr_model = Ridge(alpha=ALPHA)
expr_model.fit(X_train, Y_expr)
print("  Done")

# ---------- Metagene predictor (PCA fit on train expression) ----------
print("\n-- Fitting metagene predictor (PCA on train) --")
scaler_cp = StandardScaler()
pca_cp    = PCA(n_components=N_PCA, random_state=42)
Y_expr_sc = scaler_cp.fit_transform(Y_expr)
Y_meta    = pca_cp.fit_transform(Y_expr_sc)   # (n_train, 50)
print(f"  PCA explains {pca_cp.explained_variance_ratio_.sum():.3f} variance")

meta_model = Ridge(alpha=ALPHA)
meta_model.fit(X_train, Y_meta)
print("  Done")

# ---------- AIDO embedding predictor ----------
print("\n-- Fitting AIDO embedding predictor --")
print("  Loading trt_cp_smiles_qc_aido_cell_3m_embeddings.csv...")
emb_df = pd.read_csv(DATA_DIR / "trt_cp_smiles_qc_aido_cell_3m_embeddings.csv")
emb_df = emb_df[~emb_df["canonical_smiles"].isin(BAD_SMILES)]
emb_df = emb_df[emb_df["canonical_smiles"].notna()].copy()
emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
emb_df = emb_df.merge(cp_split[["inst_id", "split"]], on="inst_id", how="left")

train_emb_by_drug = (
    emb_df[emb_df["split"] == "train"]
    .groupby("pert_id")[emb_cols]
    .mean()
    .reset_index()
)
train_emb_aligned = train_aligned[["pert_id", "morgan"]].merge(
    train_emb_by_drug, on="pert_id", how="inner"
).reset_index(drop=True)
X_train_emb = np.stack(train_emb_aligned["morgan"].values)
Y_emb       = train_emb_aligned[emb_cols].values
print(f"  Training matrix: X={X_train_emb.shape}, Y_emb={Y_emb.shape}")

emb_model = Ridge(alpha=ALPHA)
emb_model.fit(X_train_emb, Y_emb)
print("  Done")
del emb_df, train_emb_by_drug, train_emb_aligned, X_train_emb, Y_emb

# ---------- Predict for ALL drugs ----------
print("\n-- Generating predictions for all drugs --")
X_all        = np.stack(drug_info["morgan"].values)  # (n_all, 2048)
all_pert_ids = drug_info["pert_id"].values
all_smiles   = drug_info["smiles"].values

pred_expr_arr = expr_model.predict(X_all)    # (n_drugs, 977)
pred_meta_arr = meta_model.predict(X_all)    # (n_drugs, 50)
pred_emb_arr  = emb_model.predict(X_all)     # (n_drugs, 128)

cp_pred_expr = pd.DataFrame(pred_expr_arr, columns=[f"gene_{c}" for c in gene_cols])
cp_pred_expr.insert(0, "pert_id", all_pert_ids)
cp_pred_expr.insert(1, "smiles",  all_smiles)

cp_pred_meta = pd.DataFrame(pred_meta_arr, columns=[f"metagene_{i}" for i in range(N_PCA)])
cp_pred_meta.insert(0, "pert_id", all_pert_ids)
cp_pred_meta.insert(1, "smiles",  all_smiles)

cp_pred_emb = pd.DataFrame(pred_emb_arr, columns=emb_cols)
cp_pred_emb.insert(0, "pert_id", all_pert_ids)
cp_pred_emb.insert(1, "smiles",  all_smiles)

cp_pred_expr.to_csv(OUT_DIR / "cp_pred_expression.csv",      index=False)
cp_pred_meta.to_csv(OUT_DIR / "cp_pred_metagenes.csv",       index=False)
cp_pred_emb.to_csv(OUT_DIR / "cp_pred_aido_embeddings.csv",  index=False)
print(f"  Saved cp_pred_expression.csv      {cp_pred_expr.shape}")
print(f"  Saved cp_pred_metagenes.csv       {cp_pred_meta.shape}")
print(f"  Saved cp_pred_aido_embeddings.csv {cp_pred_emb.shape}")
del cp_df, train_expr_by_drug, train_aligned, X_train, X_all


# ==============================================================================
# Part 2: Gene perturbation (trt_sh) representations
# ==============================================================================

print("\n" + "=" * 70)
print("Part 2: Building gene perturbation representations (trt_sh)")
print("=" * 70)

print("Loading trt_sh_genes_qc.csv...")
sh_df    = pd.read_csv(DATA_DIR / "trt_sh_genes_qc.csv", low_memory=False)
sh_genes = [c for c in sh_df.columns if c not in META_COLS_SH]
sh_df    = sh_df[sh_df["ensembl_id"].notna()].copy()
print(f"  {len(sh_df)} instances, {len(sh_genes)} gene columns")

sh_split = pd.read_csv(SPLIT_DIR / "trt_sh_split_map.csv")
print(f"  Splits: {sh_split['split'].value_counts().to_dict()}")
sh_df = sh_df.merge(sh_split[["inst_id", "split"]], on="inst_id", how="left")

# Train-split gene-level aggregation (used for PCA fitting)
train_sh_by_gene = (
    sh_df[sh_df["split"] == "train"]
    .groupby("ensembl_id")[sh_genes]
    .mean()
    .reset_index()
)
print(f"  Train genes: {len(train_sh_by_gene)}")

# Fit scaler and PCA on train genes only
sh_scaler = StandardScaler()
sh_pca    = PCA(n_components=N_PCA, random_state=42)
train_sc  = sh_scaler.fit_transform(train_sh_by_gene[sh_genes].values)
sh_pca.fit(train_sc)
print(f"  PCA explains {sh_pca.explained_variance_ratio_.sum():.3f} variance")

# Aggregate ALL genes (train + test) using actual measured data
all_sh_by_gene = (
    sh_df.groupby("ensembl_id")[sh_genes]
    .mean()
    .reset_index()
)
print(f"  Total genes: {len(all_sh_by_gene)}")
all_gene_ids = all_sh_by_gene["ensembl_id"].values
all_expr_arr = all_sh_by_gene[sh_genes].values

# Metagenes via train-fit scaler + PCA
all_meta_arr = sh_pca.transform(sh_scaler.transform(all_expr_arr))   # (n_genes, 50)

# AIDO embeddings
print("Loading trt_sh_genes_qc_aido_cell_3m_embeddings.csv...")
sh_emb_df = pd.read_csv(DATA_DIR / "trt_sh_genes_qc_aido_cell_3m_embeddings.csv", low_memory=False)
sh_emb_df = sh_emb_df[sh_emb_df["ensembl_id"].notna()].copy()
sh_emb_cols = [c for c in sh_emb_df.columns if c.startswith("emb_")]
sh_emb_by_gene = (
    sh_emb_df.groupby("ensembl_id")[sh_emb_cols]
    .mean()
    .reset_index()
    .rename(columns={"ensembl_id": "targetId"})
)

# Build output DataFrames
sh_repr_expr = pd.DataFrame(all_expr_arr, columns=[f"gene_{c}" for c in sh_genes])
sh_repr_expr.insert(0, "targetId", all_gene_ids)

sh_repr_meta = pd.DataFrame(all_meta_arr, columns=[f"metagene_{i}" for i in range(N_PCA)])
sh_repr_meta.insert(0, "targetId", all_gene_ids)

sh_repr_expr.to_csv(OUT_DIR / "sh_repr_expression.csv",      index=False)
sh_repr_meta.to_csv(OUT_DIR / "sh_repr_metagenes.csv",       index=False)
sh_emb_by_gene.to_csv(OUT_DIR / "sh_repr_aido_embeddings.csv", index=False)
print(f"  Saved sh_repr_expression.csv      {sh_repr_expr.shape}")
print(f"  Saved sh_repr_metagenes.csv       {sh_repr_meta.shape}")
print(f"  Saved sh_repr_aido_embeddings.csv {sh_emb_by_gene.shape}")

print("\n" + "=" * 70)
print(f"Done. All representations saved to: {OUT_DIR}")
print("=" * 70)
