"""
Bootstrap confidence intervals and all-pairs significance tests for sm_cohesion methods,
including SPRINT as a baseline.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from lightning import seed_everything

from contextpert.utils import canonicalize_smiles

SEED = 10
seed_everything(SEED, workers=True)

DATA_DIR   = Path(os.environ["CONTEXTPERT_DATA_DIR"])
MODEL_DIR  = DATA_DIR / "cellvs_molecule_networks" / "chemberta_model_outputs"
SPRINT_DIR = DATA_DIR / "sprint"
K_LIST = [1, 5, 10, 25, 50]
N_BOOTSTRAP = 10_000
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Core: per-query evaluation (returns raw score arrays, not aggregated means)
# ---------------------------------------------------------------------------

def _get_per_query_scores(pred_df, target_df, k_list=K_LIST):
    """Returns dict mapping metric name → np.array of per-query scores."""
    pred_df = pred_df.copy()
    target_df = target_df.copy()
    pred_df["smiles"] = pred_df["smiles"].apply(canonicalize_smiles)
    target_df["smiles"] = target_df["smiles"].apply(canonicalize_smiles)

    repr_cols = [c for c in pred_df.columns if c != "smiles"]
    merged = pred_df.merge(target_df, on="smiles", how="inner")

    X = merged[repr_cols].values.astype(np.float32)
    target_sig_arr = merged["targets"].values
    disease_arr = merged["diseaseId"].values

    hits = {k: [] for k in k_list}
    rrs  = {k: [] for k in k_list}

    for target_sig in np.unique(target_sig_arr):
        mask = target_sig_arr == target_sig
        gal_idx = np.where(~mask)[0]
        qry_idx = np.where(mask)[0]
        if len(gal_idx) < max(k_list):
            continue

        max_k = min(max(k_list), len(gal_idx))
        nn = NearestNeighbors(n_neighbors=max_k, metric="euclidean", algorithm="brute")
        nn.fit(X[gal_idx])
        _, nn_idx = nn.kneighbors(X[qry_idx])

        for i in range(len(qry_idx)):
            q_disease = disease_arr[qry_idx[i]]
            neighbor_diseases = disease_arr[gal_idx[nn_idx[i]]]
            for k in k_list:
                top_k = neighbor_diseases[:k]
                hits[k].append(int(q_disease in top_k))
                matches = np.where(top_k == q_disease)[0]
                rrs[k].append(1.0 / (matches[0] + 1) if len(matches) > 0 else 0.0)

    return {
        **{f"hits@{k}": np.array(hits[k]) for k in k_list},
        **{f"mrr@{k}":  np.array(rrs[k])  for k in k_list},
    }


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(scores, n=N_BOOTSTRAP, alpha=ALPHA, rng=None):
    """Returns (mean, lower, upper) via percentile bootstrap."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n_q = len(scores)
    boot = rng.integers(0, n_q, size=(n, n_q))
    means = scores[boot].mean(axis=1)
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(scores.mean()), float(lo), float(hi)


def paired_pvalue(scores_a, scores_b, n=N_BOOTSTRAP, rng=None):
    """One-sided p-value: P(scores_a <= scores_b) under bootstrap null.
    Small p-value means a is significantly better than b."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    assert len(scores_a) == len(scores_b)
    n_q = len(scores_a)
    idx = rng.integers(0, n_q, size=(n, n_q))
    diff = scores_a[idx].mean(axis=1) - scores_b[idx].mean(axis=1)
    p = float(np.mean(diff <= 0))
    return p


# ---------------------------------------------------------------------------
# Build pred_dfs for all 6 methods
# ---------------------------------------------------------------------------

print("=" * 70)
print("Loading data...")
print("=" * 70)

# Reference labels
ref_path = DATA_DIR / "opentargets/disease_drug_triples_csv/disease_drug_triples_lincs.csv"
target_df = pd.read_csv(ref_path)[["smiles", "targets", "diseaseId"]].drop_duplicates()
ref_drugs = set(target_df["smiles"].apply(canonicalize_smiles))


def filter_to_ref(df):
    df = df.copy()
    df["smiles"] = df["smiles"].apply(canonicalize_smiles)
    return df[df["smiles"].isin(ref_drugs)].copy()


# --- Random ---
print("Building: random")
rng0 = np.random.default_rng(0)
n_d = len(ref_drugs)
smiles_list = list(ref_drugs)
vecs = rng0.standard_normal((n_d, 100)).astype(np.float32)
pred_random = pd.DataFrame(vecs, columns=[f"dim_{i}" for i in range(100)])
pred_random["smiles"] = smiles_list
pred_random = filter_to_ref(pred_random)

# --- Morgan ---
print("Building: morgan")
disease_drug_df = pd.read_csv(DATA_DIR / "opentargets/disease_drug_triples_csv/disease_drug_triples.csv")
drug_smiles_list = disease_drug_df["smiles"].unique().tolist()
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps, valid_sm = [], []
for smi in drug_smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        arr = np.zeros(2048, dtype=np.float32)
        fp = morgan_gen.GetFingerprint(mol)
        for i in range(2048): arr[i] = fp[i]
        fps.append(arr)
        valid_sm.append(smi)
fps_arr = np.array(fps)
pred_morgan = pd.DataFrame(fps_arr, columns=[f"dim_{i}" for i in range(2048)])
pred_morgan["smiles"] = valid_sm
pred_morgan = filter_to_ref(pred_morgan)

# --- Expression + Metagenes ---
print("Building: expression / metagenes  (loading ~1.8 GB CSV...)")
metadata_cols = ["inst_id", "cell_id", "pert_id", "pert_type", "pert_dose",
                 "pert_dose_unit", "pert_time", "sig_id", "distil_cc_q75",
                 "pct_self_rank_q25", "canonical_smiles", "inchi_key"]
lincs_df = pd.read_csv(DATA_DIR / "trt_cp_smiles_qc.csv")
bad_smiles = ["-666", "restricted"]
lincs_df = lincs_df[~lincs_df["canonical_smiles"].isin(bad_smiles)].copy()
lincs_df = lincs_df[lincs_df["canonical_smiles"].notna()].copy()
gene_cols = [c for c in lincs_df.columns if c not in metadata_cols]

expr_by_brd = (
    lincs_df.groupby("pert_id")[gene_cols + ["canonical_smiles"]]
    .agg({**{c: "mean" for c in gene_cols}, "canonical_smiles": "first"})
    .reset_index()
)

def safe_canon(s):
    try: return canonicalize_smiles(s)
    except: return None

expr_by_brd["smiles"] = expr_by_brd["canonical_smiles"].apply(safe_canon)
expr_by_brd = expr_by_brd[expr_by_brd["smiles"].notna()].copy()

pred_expr = pd.DataFrame({"smiles": expr_by_brd["smiles"].values})
for c in gene_cols:
    pred_expr[f"gene_{c}"] = expr_by_brd[c].values
pred_expr = filter_to_ref(pred_expr)

# Metagenes: PCA-50 of expression
print("Building: metagenes (PCA)")
scaler = StandardScaler()
expr_scaled = scaler.fit_transform(expr_by_brd[gene_cols].values)
pca = PCA(n_components=50, random_state=42)
expr_pca = pca.fit_transform(expr_scaled)
pred_meta = pd.DataFrame(expr_pca, columns=[f"metagene_{i}" for i in range(50)])
pred_meta["smiles"] = expr_by_brd["smiles"].values
pred_meta = filter_to_ref(pred_meta)

# --- AIDO Cell 3M Embeddings ---
print("Building: embedding_3m")
emb_df = pd.read_csv(DATA_DIR / "trt_cp_smiles_qc_aido_cell_3m_embeddings.csv")
emb_df = emb_df[~emb_df["canonical_smiles"].isin(bad_smiles)].copy()
emb_df = emb_df[emb_df["canonical_smiles"].notna()].copy()
emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
emb_by_brd = (
    emb_df.groupby("pert_id")[emb_cols + ["canonical_smiles"]]
    .agg({**{c: "mean" for c in emb_cols}, "canonical_smiles": "first"})
    .reset_index()
)
emb_by_brd["smiles"] = emb_by_brd["canonical_smiles"].apply(safe_canon)
emb_by_brd = emb_by_brd[emb_by_brd["smiles"].notna()].copy()
pred_emb3m = pd.DataFrame({"smiles": emb_by_brd["smiles"].values})
for c in emb_cols:
    pred_emb3m[c] = emb_by_brd[c].values
pred_emb3m = filter_to_ref(pred_emb3m)

# --- Networks (ChemBERTa) ---
print("Building: networks")
betas = np.load(MODEL_DIR / "full_dataset_betas.npy")
mus   = np.load(MODEL_DIR / "full_dataset_mus.npy")
meta_csv = pd.read_csv(MODEL_DIR / "full_dataset_predictions.csv")
n_x = mus.shape[-1]
idx_upper = np.triu_indices(n_x, k=1)
bsq_raw = betas[:, idx_upper[0], idx_upper[1]] ** 2
mus_raw  = mus[:,   idx_upper[0], idx_upper[1]]
smiles_col = meta_csv["canonical_smiles"].apply(
    lambda s: canonicalize_smiles(s) if pd.notna(s) and s not in ("-666", "restricted") else None
)

def drug_mean(arr, prefix):
    cols = [f"{prefix}_{i}" for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=cols)
    df["smiles"] = smiles_col.values
    return df.dropna(subset=["smiles"]).groupby("smiles")[cols].mean().reset_index(), cols

drug_bsq, b_cols = drug_mean(bsq_raw, "b")
drug_mus, m_cols = drug_mean(mus_raw,  "m")
net_merged = drug_bsq.merge(drug_mus, on="smiles")

def std_norm(x):
    s = x.std(axis=0, keepdims=True)
    return x / np.where(s == 0, 1, s)

arr = np.hstack([std_norm(net_merged[b_cols].values),
                 std_norm(net_merged[m_cols].values)])
pred_nets = pd.DataFrame(arr, columns=[f"f_{i}" for i in range(arr.shape[1])])
pred_nets["smiles"] = net_merged["smiles"].values
pred_nets = filter_to_ref(pred_nets)

# --- SPRINT ---
print("Building: SPRINT")
sprint_drugs   = pd.read_csv(SPRINT_DIR / "drugs.csv")
sprint_drug_emb = np.load(SPRINT_DIR / "drug_embeddings.npy")
sprint_drugs["smiles_canonical"] = sprint_drugs["SMILES"].apply(safe_canon)
valid = sprint_drugs["smiles_canonical"].notna()
sprint_drugs  = sprint_drugs[valid].copy()
sprint_drug_emb = sprint_drug_emb[valid.values]
sprint_emb_dim  = sprint_drug_emb.shape[1]
pred_sprint = pd.DataFrame(sprint_drug_emb, columns=[f"emb_{i}" for i in range(sprint_emb_dim)])
pred_sprint["smiles"] = sprint_drugs["smiles_canonical"].values
pred_sprint = filter_to_ref(pred_sprint)


# ---------------------------------------------------------------------------
# Run per-query evaluation for all methods
# ---------------------------------------------------------------------------

methods = {
    "Networks":     pred_nets,
    "Expression":   pred_expr,
    "Metagenes":    pred_meta,
    "Embedding-3M": pred_emb3m,
    "Morgan":       pred_morgan,
    "SPRINT":       pred_sprint,
    "Random":       pred_random,
}

print("\nRunning per-query evaluation...")
per_query = {}
for name, pred_df in methods.items():
    print(f"  {name}...")
    per_query[name] = _get_per_query_scores(pred_df, target_df)

n_queries = len(next(iter(per_query.values()))["hits@1"])
print(f"\n{n_queries} queries total.")


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

rng = np.random.default_rng(SEED)

print("\n" + "=" * 70)
print("BOOTSTRAP 95% CIs  (n_bootstrap={:,})".format(N_BOOTSTRAP))
print("=" * 70)

ci_results = {}  # method -> metric -> (mean, lo, hi)
for name in methods:
    ci_results[name] = {}
    for metric in [f"hits@{k}" for k in K_LIST] + [f"mrr@{k}" for k in K_LIST]:
        scores = per_query[name][metric]
        mean, lo, hi = bootstrap_ci(scores, rng=rng)
        ci_results[name][metric] = (mean, lo, hi)

# Print table for Hits@k
for metric in [f"hits@{k}" for k in K_LIST]:
    print(f"\n{metric.upper()}")
    print(f"  {'Method':<15} {'Mean':>6}  {'95% CI':>20}")
    print(f"  {'-'*45}")
    for name in methods:
        mean, lo, hi = ci_results[name][metric]
        print(f"  {name:<15} {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

for metric in [f"mrr@{k}" for k in K_LIST]:
    print(f"\n{metric.upper()}")
    print(f"  {'Method':<15} {'Mean':>6}  {'95% CI':>20}")
    print(f"  {'-'*45}")
    for name in methods:
        mean, lo, hi = ci_results[name][metric]
        print(f"  {name:<15} {mean:.4f}  [{lo:.4f}, {hi:.4f}]")


# ---------------------------------------------------------------------------
# All-pairs paired bootstrap significance tests
# ---------------------------------------------------------------------------

method_names = list(methods.keys())

print("\n" + "=" * 70)
print("ALL-PAIRS PAIRED BOOTSTRAP P-VALUES")
print("(one-sided: p = P(row <= col) — small p means row significantly beats col)")
print("=" * 70)

for metric in [f"hits@{k}" for k in K_LIST] + [f"mrr@{k}" for k in K_LIST]:
    print(f"\n{metric.upper()}")
    # Header
    col_w = 13
    header = f"  {'':16}" + "".join(f"{n:>{col_w}}" for n in method_names)
    print(header)
    print("  " + "-" * (16 + col_w * len(method_names)))
    for a in method_names:
        row = f"  {a:<16}"
        for b in method_names:
            if a == b:
                row += f"{'—':>{col_w}}"
            else:
                p = paired_pvalue(per_query[a][metric], per_query[b][metric], rng=rng)
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                row += f"{p:.3f}{sig:>{col_w - 5}}"
        print(row)

print("\n* p<0.05   ** p<0.01  (row beats col)")
print("=" * 70)
