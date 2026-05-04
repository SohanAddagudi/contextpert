#!/usr/bin/env python
"""
Drug-target submission for the jointly-trained network.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from contextpert import submit_drug_target_mapping
from contextpert.utils import canonicalize_smiles

DATA_DIR  = os.environ['CONTEXTPERT_DATA_DIR']
MODEL_DIR = Path(os.environ.get(
    'MODEL_DIR',
    str(Path(__file__).parent / 'joint_model_outputs'),
))


def std_norm(x):
    s = x.std(axis=0, keepdims=True)
    return x / np.where(s == 0, 1, s)


print('=' * 80)
print('DRUG-TARGET EVALUATION (joint network, corr² ⊕ μ)')
print(f'Model dir: {MODEL_DIR.resolve()}')
print('=' * 80)

CORRS_PATH = MODEL_DIR / 'full_dataset_correlations.npy'
MUS_PATH   = MODEL_DIR / 'full_dataset_mus.npy'
PREDS_CSV  = MODEL_DIR / 'full_dataset_predictions.csv'

print(f'\nLoading correlations from: {CORRS_PATH}')
corrs_all = np.load(CORRS_PATH, mmap_mode='r')
print(f'Loading mus from:          {MUS_PATH}')
mus_all   = np.load(MUS_PATH,   mmap_mode='r')
print(f'Loading metadata from:     {PREDS_CSV}')
meta = pd.read_csv(PREDS_CSV)
print(f'  {len(meta):,} signatures | corrs {corrs_all.shape} | mus {mus_all.shape}')

n_x = mus_all.shape[-1]
idx_upper = np.triu_indices(n_x, k=1)


# -----------------------------------------------------------------------------
# Drugs (trt_cp)
# -----------------------------------------------------------------------------
print('\n' + '=' * 80)
print('DRUGS (trt_cp slice)')
print('=' * 80)

mask_cp = (meta['pert_type'] == 'trt_cp').to_numpy()
meta_cp  = meta.loc[mask_cp].reset_index(drop=True)
corrs_cp = np.asarray(corrs_all[mask_cp])
mus_cp   = np.asarray(mus_all[mask_cp])
print(f'  trt_cp signatures: {len(meta_cp):,}')

cp_bsq    = corrs_cp[:, idx_upper[0], idx_upper[1]] ** 2
cp_mus_ut = mus_cp  [:, idx_upper[0], idx_upper[1]]
nf = cp_bsq.shape[1]
b_cols = [f'b_{i}' for i in range(nf)]
m_cols = [f'm_{i}' for i in range(nf)]

smiles_col = meta_cp['canonical_smiles'].apply(
    lambda s: canonicalize_smiles(s) if pd.notna(s) and s not in ('-666', 'restricted') else None
)

print('  Aggregating by SMILES...')
df_b = pd.DataFrame(cp_bsq,    columns=b_cols); df_b['smiles'] = smiles_col.values
df_m = pd.DataFrame(cp_mus_ut, columns=m_cols); df_m['smiles'] = smiles_col.values
drug_b = df_b.dropna(subset=['smiles']).groupby('smiles')[b_cols].mean().reset_index()
drug_m = df_m.dropna(subset=['smiles']).groupby('smiles')[m_cols].mean().reset_index()
merged_cp = drug_b.merge(drug_m, on='smiles')
print(f'  unique compounds: {len(merged_cp):,}')

arr_cp = np.hstack([std_norm(merged_cp[b_cols].values), std_norm(merged_cp[m_cols].values)])
f_cols = [f'f_{i}' for i in range(arr_cp.shape[1])]
drug_preds = pd.DataFrame(arr_cp, columns=f_cols)
drug_preds['smiles'] = merged_cp['smiles'].values
print(f'  drug_preds shape: {drug_preds.shape}')
del corrs_cp, mus_cp, df_b, df_m, drug_b, drug_m, merged_cp, arr_cp


# -----------------------------------------------------------------------------
# Targets (trt_sh)
# -----------------------------------------------------------------------------
print('\n' + '=' * 80)
print('TARGETS (trt_sh slice)')
print('=' * 80)

mask_sh = (meta['pert_type'] == 'trt_sh').to_numpy()
meta_sh  = meta.loc[mask_sh].reset_index(drop=True)
corrs_sh = np.asarray(corrs_all[mask_sh])
mus_sh   = np.asarray(mus_all[mask_sh])
print(f'  trt_sh signatures: {len(meta_sh):,}')

sh_bsq    = corrs_sh[:, idx_upper[0], idx_upper[1]] ** 2
sh_mus_ut = mus_sh  [:, idx_upper[0], idx_upper[1]]
sh_nf = sh_bsq.shape[1]
sb_cols = [f'b_{i}' for i in range(sh_nf)]
sm_cols = [f'm_{i}' for i in range(sh_nf)]

# ensembl_id was stored in the joint predictions CSV at training time
ensembl_ids = meta_sh['ensembl_id'].values
print(f'  signatures with ensembl_id: {pd.notna(ensembl_ids).sum():,}')
print(f'  unique target genes:        {pd.Series(ensembl_ids).nunique():,}')

print('  Aggregating by ensembl_id...')
df_sb = pd.DataFrame(sh_bsq,    columns=sb_cols); df_sb['ensembl_id'] = ensembl_ids
df_sm = pd.DataFrame(sh_mus_ut, columns=sm_cols); df_sm['ensembl_id'] = ensembl_ids
df_sb = df_sb.dropna(subset=['ensembl_id'])
df_sm = df_sm.dropna(subset=['ensembl_id'])
tgt_b = df_sb.groupby('ensembl_id')[sb_cols].mean().reset_index()
tgt_m = df_sm.groupby('ensembl_id')[sm_cols].mean().reset_index()
merged_sh = tgt_b.merge(tgt_m, on='ensembl_id')
print(f'  unique targets: {len(merged_sh):,}')

arr_sh = np.hstack([std_norm(merged_sh[sb_cols].values), std_norm(merged_sh[sm_cols].values)])
sf_cols = [f'f_{i}' for i in range(arr_sh.shape[1])]
target_preds = pd.DataFrame(arr_sh, columns=sf_cols)
target_preds['targetId'] = merged_sh['ensembl_id'].values
print(f'  target_preds shape: {target_preds.shape}')
del corrs_sh, mus_sh, df_sb, df_sm, tgt_b, tgt_m, merged_sh, arr_sh


# -----------------------------------------------------------------------------
# Batch correction sweep + evaluation
# -----------------------------------------------------------------------------
print('\n' + '=' * 80)
print('RUNNING DRUG-TARGET MAPPING (batch-correction sweep over n_pcs in 0..3)')
print('=' * 80)

results = None
best_n_pcs = 0

for n_pcs in range(4):
    print(f'\n--- n_pcs = {n_pcs} ---')
    drug_ids   = drug_preds['smiles'].copy()
    target_ids = target_preds['targetId'].copy()

    n_drugs = len(drug_preds)
    drug_feature_cols   = [c for c in drug_preds.columns   if c != 'smiles']
    target_feature_cols = [c for c in target_preds.columns if c != 'targetId']

    X_drug   = drug_preds[drug_feature_cols].values
    X_target = target_preds[target_feature_cols].values
    X = np.vstack([X_drug, X_target])

    if n_pcs > 0:
        pca = PCA(n_components=n_pcs)
        pca.fit(X)
        X_corrected = X - pca.transform(X) @ pca.components_
        for i, ev in enumerate(pca.explained_variance_ratio_):
            print(f'    PC{i}: {ev:.4f}')
        print(f'    total removed: {pca.explained_variance_ratio_.sum():.4f}')
    else:
        X_corrected = X

    drug_preds_pca   = pd.DataFrame(X_corrected[:n_drugs],   columns=drug_feature_cols)
    drug_preds_pca.insert(0, 'smiles', drug_ids.values)
    target_preds_pca = pd.DataFrame(X_corrected[n_drugs:],   columns=target_feature_cols)
    target_preds_pca.insert(0, 'targetId', target_ids.values)

    results_tmp = submit_drug_target_mapping(drug_preds_pca, target_preds_pca)

    if not results or results_tmp.get('auroc') > results.get('auroc'):
        results = results_tmp
        best_n_pcs = n_pcs

print(f'\nBest result: n_pcs = {best_n_pcs}')


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print('\n' + '=' * 80)
print('FINAL METRICS')
print('=' * 80)

k_list = [1, 5, 10, 50]
print(f"\nDrug -> Target Retrieval ({results.get('drug_to_target_queries', 0)} queries):")
for k in k_list:
    print(f'  k={k}:')
    print(f"    Precision@{k}: {results.get(f'drug_to_target_precision@{k}', 0):.4f}")
    print(f"    Hits@{k}:    {results.get(f'drug_to_target_recall@{k}',    0):.4f}")
    print(f"    MRR@{k}:       {results.get(f'drug_to_target_mrr@{k}',     0):.4f}")

print(f"\nTarget -> Drug Retrieval ({results.get('target_to_drug_queries', 0)} queries):")
for k in k_list:
    print(f'  k={k}:')
    print(f"    Precision@{k}: {results.get(f'target_to_drug_precision@{k}', 0):.4f}")
    print(f"    Hits@{k}:    {results.get(f'target_to_drug_recall@{k}',    0):.4f}")
    print(f"    MRR@{k}:       {results.get(f'target_to_drug_mrr@{k}',     0):.4f}")

print(f"\nGraph-Based Metrics:")
print(f"  AUROC:     {results.get('auroc', 0):.4f}")
print(f"  AUPRC:     {results.get('auprc', 0):.4f}")
print(f"  Positives: {results.get('n_positives', 0):,} / {results.get('n_total_pairs', 0):,}")
print('=' * 80)
