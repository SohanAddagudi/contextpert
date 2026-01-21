import os
import json
import time
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from contextpert import submit_drug_disease_cohesion
from contextpert.utils import canonicalize_smiles



# 1. Matrix Preprocessing (Pre-Flattening)
DO_SYMMETRIZE    = True   # If True: A = 0.5 * (A + A.T)
DO_FROB_NORM     = True   # If True: Divide matrix by its Frobenius norm

# 2. Dimensionality Reduction
DO_PCA           = True   # If True: Run PCA. If False: Use raw flattened features
PCA_DIMS         = 128    # Number of components (ignored if DO_PCA is False)

# 3. Aggregation Strategy
AGG_FN           = "median" # Options: "median", "mean", "max"

# 4. Final Vector Normalization
DO_L2_NORM       = True   # If True: L2 normalize the final aggregated vectors


DATA_DIR = os.path.join('/home/user/screening3/contextpert/data')
LINCS_META_PATH = os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv')

TRAINING_OUTPUT_PARENT_DIR = '/home/user/screening2/contextpert/multi_pert_model_outputs_sota_flags1/trt_cp'
MODEL_RESULTS_DIR = os.path.join(TRAINING_OUTPUT_PARENT_DIR)

BETA_TRAIN_PATH = os.path.join(MODEL_RESULTS_DIR, 'betas_train.npy')
BETA_TEST_PATH  = os.path.join(MODEL_RESULTS_DIR, 'betas_test.npy')

PRED_CSV_PATH   = os.path.join(MODEL_RESULTS_DIR, 'predictions.csv')
OUT_SUMMARY_CSV = os.path.join(MODEL_RESULTS_DIR, 'submission_summary_betas_only.csv')



BAD_SMILES = set(["-666", "restricted"])


def exists(path: str) -> bool:
    return path is not None and os.path.isfile(path)

def symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + np.swapaxes(mat, -1, -2))

def frob_norm(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    flat = mat.reshape(mat.shape[0], -1)
    nrm = np.linalg.norm(flat, axis=1, keepdims=True) + eps
    return mat / nrm.reshape(-1, 1, 1)

def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / nrm

def vectorize_beta(A: np.ndarray, do_sym: bool, do_frob: bool) -> np.ndarray:
    """
    Vectorizes the beta matrix based on flags.
    Note: Always extracts upper-triangular (k=1) to flatten.
    """
    if do_sym:
        A = symmetrize(A)
    
    if do_frob:
        A = frob_norm(A)

    n, p, _ = A.shape
    iu = np.triu_indices(p, k=1)  # k=1 drops diagonal
    v = A[:, iu[0], iu[1]]
    return v.astype(np.float32, copy=False)

def safe_canonicalize_list(smiles_list):
    out = []
    failed = 0
    for s in smiles_list:
        try:
            out.append(canonicalize_smiles(s))
        except Exception:
            out.append(None)
            failed += 1
    if failed:
        print(f"  Warning: {failed} SMILES failed canonicalization")
    return out

def extract_numeric_score(results_obj):
    if isinstance(results_obj, dict):
        for k in ["score", "overall", "cohesion", "metric", "auroc", "auc", "mean", "avg"]:
            if k in results_obj and isinstance(results_obj[k], (int, float, np.number)):
                return float(results_obj[k]), k
        for k, v in results_obj.items():
            if isinstance(v, (int, float, np.number)):
                return float(v), k
        return float("nan"), None

    if isinstance(results_obj, (int, float, np.number)):
        return float(results_obj), "scalar"

    for attr in ["score", "overall", "metric"]:
        if hasattr(results_obj, attr):
            v = getattr(results_obj, attr)
            if isinstance(v, (int, float, np.number)):
                return float(v), attr

    return float("nan"), None

def load_and_exactly_align_predictions(n_train: int, n_test: int) -> pd.DataFrame:
    print(f"Loading prediction metadata: {PRED_CSV_PATH}")
    pred = pd.read_csv(PRED_CSV_PATH)

    required_cols = {"inst_id", "split", "sample_idx"}
    missing = required_cols - set(pred.columns)
    if missing:
        raise RuntimeError(f"predictions.csv is missing required columns: {sorted(missing)}")

    pred_train = pred[pred["split"] == "train"].copy()
    pred_test  = pred[pred["split"] == "test"].copy()

    pred_train = pred_train.sort_values("sample_idx").reset_index(drop=True)
    pred_test  = pred_test.sort_values("sample_idx").reset_index(drop=True)

    if len(pred_train) != n_train:
        raise RuntimeError(
            f"Train alignment error: predictions.csv has {len(pred_train)} train rows "
            f"but arrays have n_train={n_train}."
        )
    if len(pred_test) != n_test:
        raise RuntimeError(
            f"Test alignment error: predictions.csv has {len(pred_test)} test rows "
            f"but arrays have n_test={n_test}."
        )

    train_idx = pred_train["sample_idx"].to_numpy()
    test_idx = pred_test["sample_idx"].to_numpy()

    if not np.array_equal(train_idx, np.arange(n_train)):
        raise RuntimeError(
            "Train sample_idx is not exactly [0..n_train-1] after sorting. "
            f"Found min={train_idx.min()}, max={train_idx.max()}, unique={len(np.unique(train_idx))}."
        )
    if not np.array_equal(test_idx, np.arange(n_test)):
        raise RuntimeError(
            "Test sample_idx is not exactly [0..n_test-1] after sorting. "
            f"Found min={test_idx.min()}, max={test_idx.max()}, unique={len(np.unique(test_idx))}."
        )

    meta_aligned = pd.concat([pred_train, pred_test], ignore_index=True)
    meta_aligned = meta_aligned.reset_index(drop=True)
    meta_aligned["row_idx"] = np.arange(len(meta_aligned), dtype=np.int64)

    print(f"Exact-aligned metadata rows: {len(meta_aligned):,} (train {len(pred_train):,} + test {len(pred_test):,})")
    return meta_aligned


# -----------------------------------------------------------------------------
# Load betas (ONLY)
# -----------------------------------------------------------------------------
if not (exists(BETA_TRAIN_PATH) and exists(BETA_TEST_PATH)):
    raise FileNotFoundError(
        "Could not find betas in MODEL_RESULTS_DIR. Checked:\n"
        f"  {BETA_TRAIN_PATH}\n"
        f"  {BETA_TEST_PATH}"
    )

print("Loading betas...")
betas_train = np.load(BETA_TRAIN_PATH).astype(np.float32, copy=False)
betas_test  = np.load(BETA_TEST_PATH).astype(np.float32, copy=False)

if betas_train.shape[1] != betas_train.shape[2]:
    raise ValueError("betas_train not square")
if betas_test.shape[1] != betas_test.shape[2]:
    raise ValueError("betas_test not square")
if betas_train.shape[1] != betas_test.shape[1]:
    raise ValueError("beta train/test p mismatch")

n_train, n_test = betas_train.shape[0], betas_test.shape[0]
p_ref = betas_train.shape[1]
n_total = int(n_train + n_test)
print(f"n_train={n_train:,}, n_test={n_test:,}, n_total={n_total:,} | p={p_ref}")


meta_aligned = load_and_exactly_align_predictions(n_train=n_train, n_test=n_test)


print(f"Loading LINCS metadata: {LINCS_META_PATH}")
lincs_meta_df = pd.read_csv(LINCS_META_PATH, usecols=['inst_id', 'pert_id', 'canonical_smiles'])


print("\nPrecomputing inst_id -> pert_id + canonical smiles (and canonicalizing)...")

base_df = meta_aligned[["inst_id", "row_idx"]].copy()
base_df = pd.merge(base_df, lincs_meta_df, on="inst_id", how="left")

base_df = base_df[base_df["canonical_smiles"].notna()].copy()
base_df = base_df[~base_df["canonical_smiles"].isin(BAD_SMILES)].copy()

base_df["smiles"] = safe_canonicalize_list(base_df["canonical_smiles"].astype(str).tolist())
base_df = base_df[pd.notna(base_df["smiles"])].copy()

keep_idx = base_df["row_idx"].to_numpy(dtype=np.int64)
base_df = base_df.reset_index(drop=True)

print(f"  Kept rows after SMILES filtering/canon: {len(base_df):,} / {n_total:,}")
print(f"  Unique pert_id: {base_df['pert_id'].nunique():,}")


# -----------------------------------------------------------------------------
# Build features from betas (Using Flags)
# -----------------------------------------------------------------------------
def build_features_from_betas() -> pd.DataFrame:
    print(f"Vectorizing... (Symmetrize={DO_SYMMETRIZE}, Frob={DO_FROB_NORM})")
    Z_tr = vectorize_beta(betas_train, do_sym=DO_SYMMETRIZE, do_frob=DO_FROB_NORM)
    Z_te = vectorize_beta(betas_test, do_sym=DO_SYMMETRIZE, do_frob=DO_FROB_NORM)
    Z_full = np.concatenate([Z_tr, Z_te], axis=0).astype(np.float32, copy=False)

    if Z_full.shape[0] != n_total:
        raise RuntimeError(f"Z_full rows {Z_full.shape[0]} != n_total {n_total}")

    Z_full = Z_full[keep_idx]

    # --- PCA BLOCK ---
    if DO_PCA:
        print(f"Running PCA with dims={PCA_DIMS}...")
        pca = PCA(n_components=min(int(PCA_DIMS), Z_full.shape[1]), random_state=0)
        Z_full = pca.fit_transform(Z_full).astype(np.float32, copy=False)
    else:
        print("Skipping PCA (using raw features)...")

    feature_cols = [f"z_{i}" for i in range(Z_full.shape[1])]
    feat_df = pd.DataFrame(Z_full, columns=feature_cols)

    work_df = pd.concat([base_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    # --- AGGREGATION BLOCK ---
    print(f"Aggregating to pert_id using: {AGG_FN}...")
    brd_df = (
        work_df
        .groupby("pert_id", sort=False)[feature_cols + ["smiles"]]
        .agg({**{c: AGG_FN for c in feature_cols}, "smiles": "first"})
        .reset_index()
    )

    # --- FINAL L2 NORM BLOCK ---
    X = brd_df[feature_cols].to_numpy(np.float32, copy=False)
    
    if DO_L2_NORM:
        print("Applying final L2 normalization...")
        X = l2_normalize(X)
    else:
        print("Skipping final L2 normalization...")

    preds = pd.DataFrame(
        np.column_stack([brd_df["smiles"].astype(object).values, X.astype(np.float32)]),
        columns=["smiles"] + feature_cols
    )
    preds["smiles"] = preds["smiles"].astype(str)
    return preds


# -----------------------------------------------------------------------------
# Run once
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STARTING (BETAS ONLY)")
print("=" * 80)

t0 = time.time()
my_preds = build_features_from_betas()
build_dt = time.time() - t0
print(f"Built preds: n_compounds={len(my_preds):,}, in {build_dt:.1f}s")

t1 = time.time()
results = submit_drug_disease_cohesion(my_preds, mode="lincs")
eval_dt = time.time() - t1

score, score_key = extract_numeric_score(results)

row = {
    "rep": "beta",
    # Config logging
    "do_sym": DO_SYMMETRIZE,
    "do_frob": DO_FROB_NORM,
    "do_pca": DO_PCA,
    "do_l2": DO_L2_NORM,
    "pca_dims": int(PCA_DIMS) if DO_PCA else "raw",
    "agg_fn": AGG_FN,
    # Results
    "score": float(score),
    "score_key": score_key,
    "n_compounds": int(len(my_preds)),
    "build_seconds": float(build_dt),
    "eval_seconds": float(eval_dt),
    "raw_results_json": json.dumps(results, default=str)[:20000],
}

pd.DataFrame([row]).to_csv(OUT_SUMMARY_CSV, index=False)

print(f"\nRESULT: score={row['score']:.6f} (key={row['score_key']}) eval_seconds={row['eval_seconds']:.1f}")
print(f"Saved summary CSV: {OUT_SUMMARY_CSV}")
