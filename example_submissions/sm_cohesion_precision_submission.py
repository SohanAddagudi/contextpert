"""DDR-Bench submission using the precision (partial correlation) representation.

The shipped network submission uses squared correlations as the dependency
block. This one inverts the predicted correlation matrix first, so the
representation is the scaled precision, i.e. the Gaussian graphical model rather
than its marginal counterpart.

Partial correlations are scale free, so the inversion needs no marginal
variances and no retraining: it reads the same saved correlation predictions the
correlation submission uses. The assembled correlation matrices are usually
indefinite when replication is thin, so we symmetrize, clip entries into range,
set a unit diagonal, and project onto the PSD cone before inverting. The script
reports how often projection was needed and how far it moved each matrix.
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
CHUNK = 2048
FLOOR = 1e-6

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])
MODEL_DIR = Path(os.environ.get(
    "MODEL_DIR", str(DATA_DIR / "cellvs_molecule_networks" / "chemberta_model_outputs")))


def to_partial_correlations(corr, floor=FLOOR):
    """Correlation matrices to partial correlations, with a validity audit.

    Chunked because the prediction stack is large and float64 intermediates
    would otherwise dominate memory.
    """
    n, p = corr.shape[0], corr.shape[-1]
    idx = np.arange(p)
    out = np.empty((n, p, p), dtype=np.float32)
    nonpd, dist, norm = [], [], []
    for start in range(0, n, CHUNK):
        sl = slice(start, min(start + CHUNK, n))
        c = np.asarray(corr[sl], dtype=np.float64)
        c = (c + np.swapaxes(c, -1, -2)) / 2.0
        c = np.clip(c, -1.0, 1.0)
        c[:, idx, idx] = 1.0
        vals, vecs = np.linalg.eigh(c)
        nonpd.append(vals[:, 0] <= 0)
        norm.append(np.linalg.norm(c, axis=(1, 2)))
        pd_c = (vecs * np.clip(vals, floor, None)[:, None, :]) @ np.swapaxes(vecs, -1, -2)
        dist.append(np.linalg.norm(pd_c - c, axis=(1, 2)))
        prec = np.linalg.inv(pd_c)
        d = np.sqrt(np.diagonal(prec, axis1=-2, axis2=-1))
        partial = -prec / (d[:, :, None] * d[:, None, :])
        partial[:, idx, idx] = 1.0
        out[sl] = partial.astype(np.float32)
    nonpd, dist, norm = map(np.concatenate, (nonpd, dist, norm))
    print(f"  indefinite before projection: {nonpd.mean() * 100:.1f}% of {n:,}")
    print(f"  median projection distance:   {np.median(dist):.3f} "
          f"({np.median(dist) / np.median(norm) * 100:.1f}% of matrix norm)")
    return out


def std_norm(x):
    s = x.std(axis=0, keepdims=True)
    return x / np.where(s == 0, 1, s)


def main():
    print(f"Model dir: {MODEL_DIR.resolve()}")
    corr = np.load(MODEL_DIR / "full_dataset_correlations.npy", mmap_mode="r")
    mus = np.load(MODEL_DIR / "full_dataset_mus.npy", mmap_mode="r")
    meta = pd.read_csv(MODEL_DIR / "full_dataset_predictions.csv")
    p = corr.shape[-1]
    iu = np.triu_indices(p, k=1)
    print(f"Loaded {len(meta):,} samples | correlations: {corr.shape}")

    print("\n=== Inverting to partial correlations ===")
    partial = to_partial_correlations(corr)

    dep = partial[:, iu[0], iu[1]] ** 2
    mus_ut = np.asarray(mus[:, iu[0], iu[1]], dtype=np.float32)
    smiles = meta["canonical_smiles"].apply(
        lambda s: canonicalize_smiles(s) if pd.notna(s) and s not in ("-666", "restricted")
        else None)

    b_cols = [f"b_{i}" for i in range(dep.shape[1])]
    m_cols = [f"m_{i}" for i in range(mus_ut.shape[1])]
    db = pd.DataFrame(dep, columns=b_cols); db["smiles"] = smiles.values
    dm = pd.DataFrame(mus_ut, columns=m_cols); dm["smiles"] = smiles.values
    merged = (db.dropna(subset=["smiles"]).groupby("smiles")[b_cols].mean().reset_index()
              .merge(dm.dropna(subset=["smiles"]).groupby("smiles")[m_cols].mean().reset_index(),
                     on="smiles"))
    arr = np.hstack([std_norm(merged[b_cols].values), std_norm(merged[m_cols].values)])
    rep = pd.DataFrame(arr, columns=[f"f_{i}" for i in range(arr.shape[1])])
    rep["smiles"] = merged["smiles"].values
    print(f"\nDrug representations: {len(rep)} drugs x {arr.shape[1]} features")

    print("\n=== Evaluating DDR-Bench ===")
    results = submit_drug_disease_cohesion(rep, mode="lincs")
    print(results)


if __name__ == "__main__":
    main()
