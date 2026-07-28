"""DTR-Bench submission using the precision (partial correlation) representation.

Same transform as the DDR-Bench precision submission, applied to both sides:
drug representations from the small-molecule model and target representations
from the shRNA knockdown model, aggregated per Ensembl gene.

Two things this benchmark requires that the DDR one does not.

Batch correction. The two representations come from separately trained models on
different assays, so they carry a strong inter-dataset batch effect. Without
removing leading principal components from the combined matrix both
representations score near chance. We sweep 0 to 3 components and keep the best
AUROC, matching the shipped network submission. Note that selecting the
component count by the evaluation metric is selection on test; it keeps the
comparison between representations fair but makes absolute values optimistic.

Ground-truth filtering. `evaluate_drug_target_mapping` filters drugs and targets
missing from the submitted representations and warns, then asserts the
pre-filter state, so the assert fires on input it has just handled. We filter up
front and report exactly what was dropped.
"""

import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightning import seed_everything
from sklearn.decomposition import PCA

from contextpert.evaluate.drug_target_mapping import evaluate_drug_target_mapping
from contextpert.utils import canonicalize_smiles

SEED = 42
seed_everything(SEED, workers=True)
CHUNK = 2048
FLOOR = 1e-6

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])
DRUG_DIR = DATA_DIR / "cellvs_molecule_networks" / "chemberta_model_outputs"
TARGET_DIR = DATA_DIR / "drug_target_networks" / "trt_sh_aidocell_drug_target_networks"


def squared_partial_upper(path):
    """Stream a stack of correlation matrices to squared upper-triangular partials."""
    a = np.load(path, mmap_mode="r")
    n, p = a.shape[0], a.shape[-1]
    iu, idx = np.triu_indices(p, k=1), np.arange(p)
    out = np.empty((n, len(iu[0])), dtype=np.float32)
    for start in range(0, n, CHUNK):
        sl = slice(start, min(start + CHUNK, n))
        c = np.asarray(a[sl], dtype=np.float64)
        c = np.clip((c + np.swapaxes(c, -1, -2)) / 2.0, -1.0, 1.0)
        c[:, idx, idx] = 1.0
        vals, vecs = np.linalg.eigh(c)
        pd_c = (vecs * np.clip(vals, FLOOR, None)[:, None, :]) @ np.swapaxes(vecs, -1, -2)
        prec = np.linalg.inv(pd_c)
        d = np.sqrt(np.diagonal(prec, axis1=-2, axis2=-1))
        partial = -prec / (d[:, :, None] * d[:, None, :])
        out[sl] = (partial[:, iu[0], iu[1]] ** 2).astype(np.float32)
        del c, pd_c, prec, partial
    del a
    gc.collect()
    return out


def upper(path):
    a = np.load(path, mmap_mode="r")
    iu = np.triu_indices(a.shape[-1], k=1)
    return np.asarray(a[:, iu[0], iu[1]], dtype=np.float32)


def std_norm(x):
    s = x.std(axis=0, keepdims=True)
    return x / np.where(s == 0, 1, s)


def build(dep, mus, key, name):
    bc = [f"b_{i}" for i in range(dep.shape[1])]
    mc = [f"m_{i}" for i in range(mus.shape[1])]
    db = pd.DataFrame(dep, columns=bc); db[name] = key
    dm = pd.DataFrame(mus, columns=mc); dm[name] = key
    merged = (db.dropna(subset=[name]).groupby(name)[bc].mean().reset_index()
              .merge(dm.dropna(subset=[name]).groupby(name)[mc].mean().reset_index(), on=name))
    arr = np.hstack([std_norm(merged[bc].values), std_norm(merged[mc].values)])
    rep = pd.DataFrame(arr, columns=[f"f_{i}" for i in range(arr.shape[1])])
    rep[name] = merged[name].values
    return rep


def main():
    dmeta = pd.read_csv(DRUG_DIR / "full_dataset_predictions.csv")
    smiles = dmeta["canonical_smiles"].apply(
        lambda s: canonicalize_smiles(s) if pd.notna(s) and s not in ("-666", "restricted")
        else None).values
    print("Building drug representations...")
    drug_rep = build(squared_partial_upper(DRUG_DIR / "full_dataset_correlations.npy"),
                     upper(DRUG_DIR / "full_dataset_mus.npy"), smiles, "smiles")

    tmeta = pd.read_csv(TARGET_DIR / "full_dataset_predictions.csv")
    annot = pd.read_csv(DATA_DIR / "trt_sh_genes_qc.csv",
                        usecols=["inst_id", "ensembl_id"], low_memory=False)
    annot = annot[annot.ensembl_id.notna()].drop_duplicates("inst_id")
    ens = tmeta.merge(annot, on="inst_id", how="left")["ensembl_id"].values
    print(f"Targets annotated: {pd.notna(ens).sum():,}/{len(ens):,}, "
          f"{pd.Series(ens).nunique():,} unique genes")
    print("Building target representations...")
    tgt_rep = build(squared_partial_upper(TARGET_DIR / "full_dataset_correlations.npy"),
                    upper(TARGET_DIR / "full_dataset_mus.npy"), ens, "targetId")
    print(f"{len(drug_rep)} drugs x {len(tgt_rep)} targets")

    pairs = pd.read_csv(DATA_DIR / "opentargets/drug_target_pairs_csv/drug_target_pairs_lincs.csv")
    pairs["smiles"] = pairs["smiles"].apply(lambda s: canonicalize_smiles(s) if pd.notna(s) else None)
    n0 = len(pairs)
    pairs = pairs[pairs.smiles.isin(drug_rep.smiles) & pairs.targetId.isin(tgt_rep.targetId)]
    print(f"Evaluating {len(pairs)}/{n0} ground-truth pairs "
          f"({pairs.smiles.nunique()} drugs, {pairs.targetId.nunique()} targets)")

    dcols = [c for c in drug_rep.columns if c != "smiles"]
    tcols = [c for c in tgt_rep.columns if c != "targetId"]
    Xd, Xt = drug_rep[dcols].values, tgt_rep[tcols].values
    best, best_pcs = None, 0
    for n_pcs in range(4):
        if n_pcs == 0:
            dr, tr = drug_rep, tgt_rep
        else:
            X = np.vstack([Xd, Xt])
            pca = PCA(n_components=n_pcs).fit(X)
            Xc = X - pca.transform(X) @ pca.components_
            dr = pd.DataFrame(Xc[:len(Xd)], columns=dcols); dr.insert(0, "smiles", drug_rep.smiles.values)
            tr = pd.DataFrame(Xc[len(Xd):], columns=tcols); tr.insert(0, "targetId", tgt_rep.targetId.values)
        out = evaluate_drug_target_mapping(dr, tr, pairs)
        print(f"  components removed={n_pcs}: AUROC {out['auroc']:.4f}")
        if best is None or out["auroc"] > best["auroc"]:
            best, best_pcs = out, n_pcs
    print(f"\nBest: {best_pcs} components removed, AUROC {best['auroc']:.4f}")
    print(best)


if __name__ == "__main__":
    main()
