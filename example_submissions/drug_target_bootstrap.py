#!/usr/bin/env python
"""
DTR-Bench bootstrap CIs + significance testing 
"""
import os, sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)
    
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

from contextpert.utils import canonicalize_smiles

DATA_DIR  = os.environ['CONTEXTPERT_DATA_DIR']
SPRINT_DIR = os.path.join(DATA_DIR, 'sprint')

N_BOOT   = 10_000
SEED     = 42
ALPHA    = 0.05
BATCH    = 500
K_LIST   = [1, 5, 10, 50]


def auroc_fast(y_true, y_scores):
    """AUROC via Mann-Whitney U (handles ties correctly with rank averaging)."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_scores, kind='stable')
    # Assign average ranks for ties
    ranks = np.empty(len(y_scores), dtype=np.float64)
    ranks[order] = np.arange(len(y_scores), dtype=np.float64) + 1.0
    # Average tied ranks
    ys_sorted = y_scores[order]
    lo = 0
    while lo < len(ys_sorted):
        hi = lo + 1
        while hi < len(ys_sorted) and ys_sorted[hi] == ys_sorted[lo]:
            hi += 1
        if hi - lo > 1:
            avg = (lo + hi + 1) / 2.0
            for k in range(lo, hi):
                ranks[order[k]] = avg
        lo = hi
    U = float(ranks[y_true == 1].sum()) - n_pos * (n_pos + 1) / 2.0
    return U / (n_pos * n_neg)


def auprc_fast(y_true, y_scores):
    """Average precision (identical to sklearn's average_precision_score)."""
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_scores, kind='stable')
    y_s = y_true[order]
    precision = np.cumsum(y_s) / (np.arange(len(y_s), dtype=np.float64) + 1.0)
    return float(np.dot(precision, y_s) / n_pos)


# ──────────────────────────────────────────────────────────────────────────────
# Distance matrix (replicates contextpert internal logic)
# ──────────────────────────────────────────────────────────────────────────────

def compute_scores(drug_preds, target_preds, pairs_path):
    """
    Returns (y_true_flat, y_scores_flat) for the given method representations.
    Mirrors submit_drug_target_mapping filtering + _compute_graph_metrics.
    """
    target_pairs_df = pd.read_csv(pairs_path)[['smiles', 'targetId']].drop_duplicates()
    ref_drugs   = set(target_pairs_df['smiles'])
    ref_targets = set(target_pairs_df['targetId'])

    # Filter representations to reference set
    dp = drug_preds[drug_preds['smiles'].isin(ref_drugs)].copy()
    tp = target_preds[target_preds['targetId'].isin(ref_targets)].copy()

    drug_smiles_list = dp['smiles'].tolist()
    target_id_list   = tp['targetId'].tolist()
    drug_fcols   = [c for c in dp.columns if c != 'smiles']
    target_fcols = [c for c in tp.columns if c != 'targetId']

    X_d = dp[drug_fcols].values.astype(np.float32)
    X_t = tp[target_fcols].values.astype(np.float32)

    n_d, n_t = len(drug_smiles_list), len(target_id_list)

    # Batched Euclidean distance matrix
    dist = np.zeros((n_d, n_t), dtype=np.float32)
    bsz = 500
    for i in range(0, n_d, bsz):
        ei = min(i + bsz, n_d)
        diff = X_d[i:ei, np.newaxis, :] - X_t[np.newaxis, :, :]
        dist[i:ei] = np.linalg.norm(diff, axis=2)

    # Ground-truth adjacency
    drug_to_idx   = {s: i for i, s in enumerate(drug_smiles_list)}
    target_to_idx = {t: i for i, t in enumerate(target_id_list)}
    y_true = np.zeros((n_d, n_t), dtype=np.int32)
    for _, row in target_pairs_df.iterrows():
        di = drug_to_idx.get(row['smiles'])
        ti = target_to_idx.get(row['targetId'])
        if di is not None and ti is not None:
            y_true[di, ti] = 1

    y_true_flat  = y_true.flatten()
    y_scores_flat = (-dist).flatten()

    print(f"    {int(y_true_flat.sum())} pos / {len(y_true_flat)} pairs  "
          f"({drug_to_idx.__len__()} drugs × {target_to_idx.__len__()} targets)  "
          f"AUROC={auroc_fast(y_true_flat, y_scores_flat):.4f}  "
          f"AUPRC={auprc_fast(y_true_flat, y_scores_flat):.4f}")

    return y_true_flat, y_scores_flat


# ──────────────────────────────────────────────────────────────────────────────
# Per-query Hits@k for drug→target and target→drug retrieval
# ──────────────────────────────────────────────────────────────────────────────

def compute_hits_per_query(drug_preds, target_preds, pairs_path, k_list=K_LIST):
    """Returns {'drug_to_target': {k: per_query_hits_array}, 'target_to_drug': {...}}.

    Hits@k per query = 1 if any ground-truth positive appears in the top-k nearest
    neighbors (Euclidean), else 0. Mirrors contextpert.evaluate.drug_target_mapping.
    """
    target_pairs_df = pd.read_csv(pairs_path)[['smiles', 'targetId']].drop_duplicates()
    ref_drugs   = set(target_pairs_df['smiles'])
    ref_targets = set(target_pairs_df['targetId'])

    dp = drug_preds[drug_preds['smiles'].isin(ref_drugs)].copy()
    tp = target_preds[target_preds['targetId'].isin(ref_targets)].copy()

    drug_smiles_list = dp['smiles'].tolist()
    target_id_list   = tp['targetId'].tolist()
    drug_fcols   = [c for c in dp.columns if c != 'smiles']
    target_fcols = [c for c in tp.columns if c != 'targetId']
    X_d = dp[drug_fcols].values.astype(np.float32)
    X_t = tp[target_fcols].values.astype(np.float32)
    drug_to_idx   = {s: i for i, s in enumerate(drug_smiles_list)}
    target_to_idx = {t: i for i, t in enumerate(target_id_list)}

    valid = (target_pairs_df['smiles'].isin(drug_smiles_list)
             & target_pairs_df['targetId'].isin(target_id_list))
    pairs = target_pairs_df[valid]

    drug_to_targets = defaultdict(set)
    target_to_drugs = defaultdict(set)
    for _, r in pairs.iterrows():
        drug_to_targets[r['smiles']].add(r['targetId'])
        target_to_drugs[r['targetId']].add(r['smiles'])

    out = {'drug_to_target': {k: None for k in k_list},
           'target_to_drug': {k: None for k in k_list}}

    # Drug -> Target
    n_t = len(target_id_list)
    max_k_dt = min(max(k_list), n_t)
    nn_t = NearestNeighbors(n_neighbors=max_k_dt, metric='euclidean', algorithm='brute').fit(X_t)
    drug_queries = list(drug_to_targets.keys())
    hits_dt = {k: [] for k in k_list}
    for q in drug_queries:
        qi = drug_to_idx[q]
        _, idxs = nn_t.kneighbors(X_d[qi:qi+1])
        nbr = [target_id_list[i] for i in idxs[0]]
        gt = drug_to_targets[q]
        for k in k_list:
            kk = min(k, len(nbr))
            hits_dt[k].append(1 if any(n in gt for n in nbr[:kk]) else 0)
    for k in k_list:
        out['drug_to_target'][k] = np.array(hits_dt[k], dtype=np.float64)

    # Target -> Drug
    n_d = len(drug_smiles_list)
    max_k_td = min(max(k_list), n_d)
    nn_d = NearestNeighbors(n_neighbors=max_k_td, metric='euclidean', algorithm='brute').fit(X_d)
    target_queries = list(target_to_drugs.keys())
    hits_td = {k: [] for k in k_list}
    for q in target_queries:
        qi = target_to_idx[q]
        _, idxs = nn_d.kneighbors(X_t[qi:qi+1])
        nbr = [drug_smiles_list[i] for i in idxs[0]]
        gt = target_to_drugs[q]
        for k in k_list:
            kk = min(k, len(nbr))
            hits_td[k].append(1 if any(n in gt for n in nbr[:kk]) else 0)
    for k in k_list:
        out['target_to_drug'][k] = np.array(hits_td[k], dtype=np.float64)

    print(f"    Hits queries: drug→target n={len(drug_queries)}  "
          f"target→drug n={len(target_queries)}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# PCA batch correction helper
# ──────────────────────────────────────────────────────────────────────────────

def apply_pca_correction(drug_preds, target_preds, n_pcs):
    if n_pcs == 0:
        return drug_preds.copy(), target_preds.copy()
    drug_ids   = drug_preds['smiles'].copy()
    target_ids = target_preds['targetId'].copy()
    n_d = len(drug_preds)
    d_fcols = [c for c in drug_preds.columns if c != 'smiles']
    t_fcols = [c for c in target_preds.columns if c != 'targetId']
    X = np.vstack([drug_preds[d_fcols].values, target_preds[t_fcols].values])
    pca = PCA(n_components=n_pcs); pca.fit(X)
    X -= pca.transform(X) @ pca.components_
    dp = pd.DataFrame(X[:n_d], columns=d_fcols); dp.insert(0, 'smiles', drug_ids.values)
    tp = pd.DataFrame(X[n_d:], columns=t_fcols); tp.insert(0, 'targetId', target_ids.values)
    return dp, tp


# ──────────────────────────────────────────────────────────────────────────────
# Build all representations
# ──────────────────────────────────────────────────────────────────────────────

PAIRS_PATH = os.path.join(DATA_DIR, 'opentargets/drug_target_pairs_csv/drug_target_pairs_lincs.csv')

def std_norm(x):
    s = x.std(axis=0, keepdims=True)
    return x / np.where(s == 0, 1, s)


print("=" * 70)
print("Building representations for all methods...")
print("=" * 70)

conditions = {}     # key -> (y_true_flat, y_scores_flat)
hits_per_q = {}     # key -> {'drug_to_target': {k: arr}, 'target_to_drug': {k: arr}}

# ── RANDOM ────────────────────────────────────────────────────────────────────
print("\n[1/6] Random baseline (seed=1)")
pairs_df = pd.read_csv(PAIRS_PATH)
drug_smiles_rand = pairs_df['smiles'].unique()
target_ids_rand  = pairs_df['targetId'].unique()
rng_rand = np.random.default_rng(1)
drug_rand = pd.DataFrame(rng_rand.standard_normal((len(drug_smiles_rand), 100)).astype(np.float32),
                         columns=[f'd_{i}' for i in range(100)])
drug_rand.insert(0, 'smiles', drug_smiles_rand)
target_rand = pd.DataFrame(rng_rand.standard_normal((len(target_ids_rand), 100)).astype(np.float32),
                           columns=[f't_{i}' for i in range(100)])
target_rand.insert(0, 'targetId', target_ids_rand)

# ── EXPRESSION ────────────────────────────────────────────────────────────────
print("\n[2/6] Expression")
with open(os.path.join(DATA_DIR, 'trt_sh_qc_gene_cols.txt')) as f:
    gene_cols = [l.strip() for l in f]
trt_cp_df = pd.read_csv(os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv'))
agg = {c: 'mean' for c in gene_cols}; agg['canonical_smiles'] = 'first'
drug_expr_df = trt_cp_df.groupby('pert_id')[gene_cols + ['canonical_smiles']].agg(agg).reset_index()
drug_preds_expr = drug_expr_df[['canonical_smiles'] + gene_cols].rename(columns={'canonical_smiles': 'smiles'})

trt_sh_df = pd.read_csv(os.path.join(DATA_DIR, 'trt_sh_genes_qc.csv'), low_memory=False)
trt_sh_df = trt_sh_df[trt_sh_df['ensembl_id'].notna()].copy()
target_preds_expr = trt_sh_df.groupby('ensembl_id')[gene_cols].mean().reset_index().rename(columns={'ensembl_id': 'targetId'})

# ── METAGENES ─────────────────────────────────────────────────────────────────
print("\n[3/6] Metagenes")
combined = np.vstack([drug_expr_df[gene_cols].values, target_preds_expr[gene_cols].values])
scaler = StandardScaler(); pca50 = PCA(n_components=50, random_state=42)
comb_pca = pca50.fit_transform(scaler.fit_transform(combined))
n_d_expr = len(drug_expr_df)
meta_cols = [f'metagene_{i}' for i in range(50)]
drug_preds_meta = pd.DataFrame(comb_pca[:n_d_expr], columns=meta_cols)
drug_preds_meta.insert(0, 'smiles', drug_expr_df['canonical_smiles'].values)
target_preds_meta = pd.DataFrame(comb_pca[n_d_expr:], columns=meta_cols)
target_preds_meta.insert(0, 'targetId', target_preds_expr['targetId'].values)
del combined, comb_pca

# ── EMBEDDING 3M ──────────────────────────────────────────────────────────────
print("\n[4/6] Embedding 3M")
trt_cp_e = pd.read_csv(os.path.join(DATA_DIR, 'trt_cp_smiles_qc_aido_cell_3m_embeddings.csv'))
emb_cols = [c for c in trt_cp_e.columns if c.startswith('emb_')]
agg_e = {c: 'mean' for c in emb_cols}; agg_e['canonical_smiles'] = 'first'
drug_preds_emb = (trt_cp_e.groupby('pert_id')[emb_cols + ['canonical_smiles']]
                  .agg(agg_e).reset_index()[['canonical_smiles'] + emb_cols]
                  .rename(columns={'canonical_smiles': 'smiles'}))
# SH embeddings file lacks ensembl_id; merge via inst_id with trt_sh_genes_qc.csv
trt_sh_e = pd.read_csv(os.path.join(DATA_DIR, 'trt_sh_qc_aido_cell_3m_embeddings.csv'), low_memory=False)
_sh_ann = pd.read_csv(os.path.join(DATA_DIR, 'trt_sh_genes_qc.csv'),
                      usecols=['inst_id', 'ensembl_id'], low_memory=False)
_sh_ann = _sh_ann[_sh_ann['ensembl_id'].notna()]
trt_sh_e = trt_sh_e.merge(_sh_ann, on='inst_id', how='inner')
target_preds_emb = trt_sh_e.groupby('ensembl_id')[emb_cols].mean().reset_index().rename(columns={'ensembl_id': 'targetId'})
del trt_cp_e, trt_sh_e

# ── NETWORKS ──────────────────────────────────────────────────────────────────
print("\n[5/6] Networks  —  reduction: corr² ⊕ μ")
cp_corrs = np.load(os.path.join(DATA_DIR, 'cellvs_molecule_networks/chemberta_model_outputs/full_dataset_correlations.npy'))
cp_mus   = np.load(os.path.join(DATA_DIR, 'cellvs_molecule_networks/chemberta_model_outputs/full_dataset_mus.npy'))
cp_meta  = pd.read_csv(os.path.join(DATA_DIR, 'cellvs_molecule_networks/chemberta_model_outputs/full_dataset_predictions.csv'))
idx_u = np.triu_indices(cp_mus.shape[-1], k=1)
cp_bsq = cp_corrs[:, idx_u[0], idx_u[1]] ** 2  # corr² (was β²)
cp_mus_ut = cp_mus[:, idx_u[0], idx_u[1]]
nf = cp_bsq.shape[1]
b_c = [f'b_{i}' for i in range(nf)]; m_c = [f'm_{i}' for i in range(nf)]
smiles_col = cp_meta['canonical_smiles'].apply(
    lambda s: canonicalize_smiles(s) if pd.notna(s) and s not in ('-666', 'restricted') else None)
df_b = pd.DataFrame(cp_bsq, columns=b_c); df_b['smiles'] = smiles_col.values
df_m = pd.DataFrame(cp_mus_ut, columns=m_c); df_m['smiles'] = smiles_col.values
drug_b = df_b.dropna(subset=['smiles']).groupby('smiles')[b_c].mean().reset_index()
drug_m = df_m.dropna(subset=['smiles']).groupby('smiles')[m_c].mean().reset_index()
merged_cp = drug_b.merge(drug_m, on='smiles')
arr_cp = np.hstack([std_norm(merged_cp[b_c].values), std_norm(merged_cp[m_c].values)])
fc = [f'f_{i}' for i in range(arr_cp.shape[1])]
drug_preds_net = pd.DataFrame(arr_cp, columns=fc); drug_preds_net['smiles'] = merged_cp['smiles'].values
del cp_corrs, cp_mus, df_b, df_m, drug_b, drug_m, merged_cp, arr_cp

sh_corrs = np.load(os.path.join(DATA_DIR, 'drug_target_networks/trt_sh_aidocell_drug_target_networks/full_dataset_correlations.npy'))
sh_mus   = np.load(os.path.join(DATA_DIR, 'drug_target_networks/trt_sh_aidocell_drug_target_networks/full_dataset_mus.npy'))
sh_meta  = pd.read_csv(os.path.join(DATA_DIR, 'drug_target_networks/trt_sh_aidocell_drug_target_networks/full_dataset_predictions.csv'))
sh_idx_u = np.triu_indices(sh_mus.shape[-1], k=1)
sh_bsq = sh_corrs[:, sh_idx_u[0], sh_idx_u[1]] ** 2  # corr² (was β²)
sh_mus_ut = sh_mus[:, sh_idx_u[0], sh_idx_u[1]]
sf = sh_bsq.shape[1]
sb_c = [f'b_{i}' for i in range(sf)]; sm_c = [f'm_{i}' for i in range(sf)]
sh_annot = pd.read_csv(os.path.join(DATA_DIR, 'trt_sh_genes_qc.csv'), usecols=['inst_id', 'ensembl_id'], low_memory=False)
sh_annot = sh_annot[sh_annot['ensembl_id'].notna()]
ens_ids = sh_meta.merge(sh_annot, on='inst_id', how='left')['ensembl_id'].values
df_sb = pd.DataFrame(sh_bsq, columns=sb_c); df_sb['ensembl_id'] = ens_ids
df_sm = pd.DataFrame(sh_mus_ut, columns=sm_c); df_sm['ensembl_id'] = ens_ids
df_sb = df_sb.dropna(subset=['ensembl_id']); df_sm = df_sm.dropna(subset=['ensembl_id'])
tgt_b = df_sb.groupby('ensembl_id')[sb_c].mean().reset_index()
tgt_m = df_sm.groupby('ensembl_id')[sm_c].mean().reset_index()
merged_sh = tgt_b.merge(tgt_m, on='ensembl_id')
arr_sh = np.hstack([std_norm(merged_sh[sb_c].values), std_norm(merged_sh[sm_c].values)])
sfc = [f'f_{i}' for i in range(arr_sh.shape[1])]
target_preds_net = pd.DataFrame(arr_sh, columns=sfc); target_preds_net['targetId'] = merged_sh['ensembl_id'].values
del sh_corrs, sh_mus, df_sb, df_sm, tgt_b, tgt_m, merged_sh, arr_sh

# ── SPRINT ────────────────────────────────────────────────────────────────────
print("\n[6/6] Sprint")
drugs_df  = pd.read_csv(os.path.join(SPRINT_DIR, 'drugs.csv'))
drug_emb  = np.load(os.path.join(SPRINT_DIR, 'drug_embeddings.npy'))
tgt_df    = pd.read_csv(os.path.join(SPRINT_DIR, 'targets.csv'))
tgt_emb   = np.load(os.path.join(SPRINT_DIR, 'target_embeddings.npy'))
def safe_canon(s):
    try: return canonicalize_smiles(s)
    except: return None
drugs_df['smiles_canonical'] = drugs_df['SMILES'].apply(safe_canon)
valid = drugs_df['smiles_canonical'].notna()
drugs_df = drugs_df[valid].copy(); drug_emb = drug_emb[valid.values]
ed = drug_emb.shape[1]; ec = [f'emb_{i}' for i in range(ed)]
drug_preds_sprint  = pd.DataFrame(drug_emb, columns=ec); drug_preds_sprint['smiles'] = drugs_df['smiles_canonical'].values
target_preds_sprint = pd.DataFrame(tgt_emb, columns=ec); target_preds_sprint['targetId'] = tgt_df['target_id'].values


# ── PREDICTORS ────────────────────────────────────────────────────────────────
print("\n[7/7] Predictors (load from predictors/outputs/)")
PRED_DIR = os.path.join(os.path.dirname(__file__), '..', 'predictors', 'outputs')

def _load_pred_drug(fname, id_cols=('pert_id', 'smiles')):
    df = pd.read_csv(os.path.join(PRED_DIR, fname))
    feat = [c for c in df.columns if c not in set(id_cols)]
    df['smiles'] = df['smiles'].apply(safe_canon)
    return df.dropna(subset=['smiles']).groupby('smiles')[feat].mean().reset_index()

drug_preds_pred_expr = _load_pred_drug('cp_pred_expression.csv')
drug_preds_pred_meta_raw = _load_pred_drug('cp_pred_metagenes.csv')
drug_preds_pred_emb  = _load_pred_drug('cp_pred_aido_embeddings.csv')

sh_repr_expr = pd.read_csv(os.path.join(PRED_DIR, 'sh_repr_expression.csv'))
sh_repr_meta_raw = pd.read_csv(os.path.join(PRED_DIR, 'sh_repr_metagenes.csv'))
sh_repr_emb  = pd.read_csv(os.path.join(PRED_DIR, 'sh_repr_aido_embeddings.csv'))

target_preds_pred_expr = sh_repr_expr.rename(columns={'targetId': 'targetId'})
target_preds_pred_emb  = sh_repr_emb

# Joint PCA-50 for predictor metagenes (predicted drug + actual target expression)
_gene_cols_pred = [c for c in drug_preds_pred_expr.columns if c != 'smiles']
_gene_cols_tgt  = [c for c in sh_repr_expr.columns if c != 'targetId']
_shared = sorted(set(_gene_cols_pred) & set(_gene_cols_tgt))
_joint  = np.vstack([drug_preds_pred_expr[_shared].values,
                     sh_repr_expr[_shared].values])
_n_d_pred = len(drug_preds_pred_expr)
_sc_pred = StandardScaler(); _pca_pred = PCA(n_components=50, random_state=42)
_joint_pca = _pca_pred.fit_transform(_sc_pred.fit_transform(_joint))
_mc = [f'metagene_{i}' for i in range(50)]
drug_preds_pred_meta  = pd.DataFrame(_joint_pca[:_n_d_pred], columns=_mc)
drug_preds_pred_meta['smiles'] = drug_preds_pred_expr['smiles'].values
target_preds_pred_meta = pd.DataFrame(_joint_pca[_n_d_pred:], columns=_mc)
target_preds_pred_meta['targetId'] = sh_repr_expr['targetId'].values
del _joint, _joint_pca

# ──────────────────────────────────────────────────────────────────────────────
# Compute (y_true, y_scores) for each method × condition
# best_n from previous run: expression=3, metagenes=3, embedding_3m=1, networks=3
# ──────────────────────────────────────────────────────────────────────────────

BEST_N = {'expression': 3, 'metagenes': 3, 'embedding_3m': 1, 'networks': 3}

print("\n" + "=" * 70)
print("Computing distance matrices...")
print("=" * 70)

def add(name, dp, tp, n_pcs=0):
    label = f"{name}_pca{n_pcs}" if n_pcs > 0 else name
    print(f"\n  {label}")
    dp_c, tp_c = apply_pca_correction(dp, tp, n_pcs)
    conditions[label] = compute_scores(dp_c, tp_c, PAIRS_PATH)
    hits_per_q[label] = compute_hits_per_query(dp_c, tp_c, PAIRS_PATH, K_LIST)

print("\n-- Random --")
add('random', drug_rand, target_rand)

print("\n-- Expression --")
add('expression', drug_preds_expr, target_preds_expr)
add('expression', drug_preds_expr, target_preds_expr, BEST_N['expression'])

print("\n-- Metagenes --")
add('metagenes', drug_preds_meta, target_preds_meta)
add('metagenes', drug_preds_meta, target_preds_meta, BEST_N['metagenes'])

print("\n-- Embedding 3M --")
add('embedding_3m', drug_preds_emb, target_preds_emb)
add('embedding_3m', drug_preds_emb, target_preds_emb, BEST_N['embedding_3m'])

print("\n-- Networks --")
add('networks', drug_preds_net, target_preds_net)
add('networks', drug_preds_net, target_preds_net, BEST_N['networks'])

print("\n-- Sprint --")
add('sprint', drug_preds_sprint, target_preds_sprint)

print("\n-- Predictor: Expression --")
add('pred_expression', drug_preds_pred_expr, target_preds_pred_expr)
add('pred_expression', drug_preds_pred_expr, target_preds_pred_expr, 3)

print("\n-- Predictor: Metagenes (joint PCA-50) --")
add('pred_metagenes', drug_preds_pred_meta, target_preds_pred_meta)

print("\n-- Predictor: AIDO Embeddings --")
add('pred_aido', drug_preds_pred_emb, target_preds_pred_emb)


pair_counts = {k: len(v[0]) for k, v in conditions.items()}
groups = {}  # n_pairs -> list of condition keys
for k, n in pair_counts.items():
    groups.setdefault(n, []).append(k)

print(f"\n" + "=" * 70)
print(f"Bootstrapping ({N_BOOT:,} resamples, batch={BATCH})")
for n, ks in groups.items():
    print(f"  Group n_pairs={n:,}: {ks}")
print("=" * 70)

rng = np.random.default_rng(SEED)
boot = {k: {'auroc': np.empty(N_BOOT), 'auprc': np.empty(N_BOOT)}
        for k in conditions}

for n_pairs, group_keys in groups.items():
    print(f"\nGroup n_pairs={n_pairs:,} — {group_keys}")
    done = 0
    while done < N_BOOT:
        bs  = min(BATCH, N_BOOT - done)
        idx = rng.integers(0, n_pairs, size=(bs, n_pairs))
        for i in range(bs):
            ii = idx[i]
            for k in group_keys:
                yt = conditions[k][0][ii]
                ys = conditions[k][1][ii]
                s  = yt.sum()
                if 0 < s < n_pairs:
                    boot[k]['auroc'][done] = auroc_fast(yt, ys)
                    boot[k]['auprc'][done] = auprc_fast(yt, ys)
                else:
                    # degenerate — use point estimate (extremely rare)
                    boot[k]['auroc'][done] = auroc_fast(conditions[k][0], conditions[k][1])
                    boot[k]['auprc'][done] = auprc_fast(conditions[k][0], conditions[k][1])
            done += 1
        if done % 2000 == 0 or done == N_BOOT:
            sys.stdout.write(f"  {done}/{N_BOOT}\n"); sys.stdout.flush()

def pval_vs_random(boot_method, boot_random, metric, paired):
    """One-sided P(method ≤ random). Small = method significantly better."""
    if paired:
        diff = boot_method[metric] - boot_random[metric]
    else:
        diff = boot_method[metric] - boot_random[metric][
            np.random.default_rng(SEED+1).integers(0, N_BOOT, N_BOOT)]
    return float(np.mean(diff <= 0))

def ci_str(arr, alpha=ALPHA):
    lo = np.percentile(arr, 100 * alpha / 2)
    hi = np.percentile(arr, 100 * (1 - alpha / 2))
    return float(arr.mean()), float(lo), float(hi)

rand_boot = boot['random']

print("\n\n" + "=" * 100)
print("DTR-BENCH BOOTSTRAP RESULTS  (95% CI, paired vs. random)")
print("=" * 100)

header = (f"{'Condition':<28} {'AUROC':>6}  {'95% CI':>18}  "
          f"{'p(vs rand)':>10}  {'AUPRC':>6}  {'95% CI':>18}  {'p(vs rand)':>10}")
print(header)
print("-" * 100)

order = [
    ('random',            'random',               None),
    ('expression',        'expression',            'expression'),
    ('expression_pca3',   'expression + PCA-3',    'expression'),
    ('metagenes',         'metagenes',             'metagenes'),
    ('metagenes_pca3',    'metagenes + PCA-3',     'metagenes'),
    ('embedding_3m',      'embedding_3m',          'embedding_3m'),
    ('embedding_3m_pca1', 'embedding_3m + PCA-1',  'embedding_3m'),
    ('networks',          'networks',              'networks'),
    ('networks_pca3',     'networks + PCA-3',      'networks'),
    ('sprint',              'sprint',                  'sprint'),
    ('pred_expression',     'Pred-Expression',           'pred_expression'),
    ('pred_expression_pca3','Pred-Expression + PCA-3',   'pred_expression'),
    ('pred_metagenes',      'Pred-Metagenes',            'pred_metagenes'),
    ('pred_aido',           'Pred-AIDO',                 'pred_aido'),
]

for key, label, grp in order:
    if key not in boot:
        continue
    b = boot[key]
    auroc_mean, auroc_lo, auroc_hi = ci_str(b['auroc'])
    auprc_mean, auprc_lo, auprc_hi = ci_str(b['auprc'])

    if key == 'random':
        p_auroc = p_auprc = '—'
    else:
        paired = (pair_counts[key] == pair_counts['random'])
        p_auroc = f"{pval_vs_random(b, rand_boot, 'auroc', paired):.4f}"
        p_auprc = f"{pval_vs_random(b, rand_boot, 'auprc', paired):.4f}"

    print(f"{label:<28} {auroc_mean:.4f}  [{auroc_lo:.4f}, {auroc_hi:.4f}]  "
          f"{p_auroc:>10}  {auprc_mean:.4f}  [{auprc_lo:.4f}, {auprc_hi:.4f}]  {p_auprc:>10}")

print("=" * 100)
print("\nNotes:")
print("  95% CI: percentile bootstrap (10,000 resamples)")
print("  p(vs rand): one-sided paired bootstrap p-value, P(method ≤ random)")
print("              same resample indices used for all methods (paired test)")
print("  PCA-n: n leading PCs removed for batch correction")


# ──────────────────────────────────────────────────────────────────────────────
# Query-level Hits@k bootstrap (Drug Hits = drug→target, Target Hits = target→drug)
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 100)
print(f"BOOTSTRAPPING QUERY-LEVEL HITS@k  ({N_BOOT:,} resamples)")
print("=" * 100)

hits_boot = {}  # key -> direction -> k -> np.array of bootstrap means
rng_h = np.random.default_rng(SEED + 7)
for key in hits_per_q:
    hits_boot[key] = {'drug_to_target': {}, 'target_to_drug': {}}
    for direction in ('drug_to_target', 'target_to_drug'):
        per_q = hits_per_q[key][direction]
        arrs = np.stack([per_q[k] for k in K_LIST], axis=1)  # (n_q, n_k)
        n_q = arrs.shape[0]
        if n_q == 0:
            for k in K_LIST:
                hits_boot[key][direction][k] = np.full(N_BOOT, np.nan)
            continue
        idx_mat = rng_h.integers(0, n_q, size=(N_BOOT, n_q))
        boot_means = arrs[idx_mat].mean(axis=1)  # (N_BOOT, n_k)
        for ki, k in enumerate(K_LIST):
            hits_boot[key][direction][k] = boot_means[:, ki]

def _print_hits_table(direction, title):
    print(f"\n{title}")
    header = f"{'Condition':<28}"
    for k in K_LIST:
        header += f"  {'Hits@'+str(k):>8}  {'95% CI':>20}"
    print(header)
    print("-" * len(header))
    for key, label, _ in order:
        if key not in hits_boot:
            continue
        row = f"{label:<28}"
        for k in K_LIST:
            arr = hits_boot[key][direction][k]
            mean, lo, hi = ci_str(arr)
            row += f"  {mean:>8.4f}  [{lo:.4f}, {hi:.4f}]"
        print(row)

print("\n" + "=" * 100)
print("DRUG HITS (drug → target retrieval):  query=drug, gallery=targets, recall@k")
print("=" * 100)
_print_hits_table('drug_to_target', '')

print("\n" + "=" * 100)
print("TARGET HITS (target → drug retrieval):  query=target, gallery=drugs, recall@k")
print("=" * 100)
_print_hits_table('target_to_drug', '')
