"""
sm_cohesion submission for the jointly-trained network.

Mirrors the Networks block of sm_cohesion_bootstrap.py (corr² ⊕ μ reduction)
applied to the trt_cp slice of the joint model's outputs.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightning import seed_everything

from contextpert import submit_drug_disease_cohesion
from contextpert.utils import canonicalize_smiles

SEED = 10
seed_everything(SEED, workers=True)

DATA_DIR  = Path(os.environ['CONTEXTPERT_DATA_DIR'])
MODEL_DIR = Path(os.environ.get(
    'MODEL_DIR',
    str(Path(__file__).parent / 'joint_model_outputs'),
))

print('=' * 70)
print(f'Model dir: {MODEL_DIR.resolve()}')
print('=' * 70)

CORRS_PATH = MODEL_DIR / 'full_dataset_correlations.npy'
MUS_PATH   = MODEL_DIR / 'full_dataset_mus.npy'
PREDS_CSV  = MODEL_DIR / 'full_dataset_predictions.csv'

print('\n=== Loading joint network predictions ===')
corrs = np.load(CORRS_PATH, mmap_mode='r')
mus   = np.load(MUS_PATH,   mmap_mode='r')
meta  = pd.read_csv(PREDS_CSV)
print(f'Loaded {len(meta):,} signatures | corrs: {corrs.shape} | mus: {mus.shape}')

# Restrict to trt_cp rows
mask = (meta['pert_type'] == 'trt_cp').to_numpy()
meta_cp = meta.loc[mask].reset_index(drop=True)
corrs_cp = np.asarray(corrs[mask])
mus_cp   = np.asarray(mus[mask])
print(f'trt_cp slice: {len(meta_cp):,} signatures')

# corr² ⊕ μ over the strict upper triangle
n_x = mus_cp.shape[-1]
idx_upper = np.triu_indices(n_x, k=1)
bsq_raw = corrs_cp[:, idx_upper[0], idx_upper[1]] ** 2
mus_raw = mus_cp[:,   idx_upper[0], idx_upper[1]]

print('\n=== Building drug representations ===')
smiles_col = meta_cp['canonical_smiles'].apply(
    lambda s: canonicalize_smiles(s) if pd.notna(s) and s not in ('-666', 'restricted') else None
)


def drug_mean(arr, prefix):
    cols = [f'{prefix}_{i}' for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=cols)
    df['smiles'] = smiles_col.values
    return df.dropna(subset=['smiles']).groupby('smiles')[cols].mean().reset_index(), cols


drug_bsq, b_cols = drug_mean(bsq_raw, 'b')
drug_mus, m_cols = drug_mean(mus_raw, 'm')
merged = drug_bsq.merge(drug_mus, on='smiles')
print(f'Drug representations: {len(merged)} drugs × {len(b_cols) + len(m_cols)} features '
      f'({len(b_cols)} corr² + {len(m_cols)} mus)')


def std_norm(x):
    s = x.std(axis=0, keepdims=True)
    return x / np.where(s == 0, 1, s)


arr = np.hstack([std_norm(merged[b_cols].values), std_norm(merged[m_cols].values)])
drug_rep = pd.DataFrame(arr, columns=[f'f_{i}' for i in range(arr.shape[1])])
drug_rep['smiles'] = merged['smiles'].values

print('\n=== Evaluating sm_cohesion ===')
results = submit_drug_disease_cohesion(drug_rep, mode='lincs')
