"""
Build ctrls.csv from the per-perturbation-type control CSVs.

Concatenates ctl_untrt.csv, ctl_vector.csv, and ctl_vehicle.csv from
$CONTEXTPERT_DATA_DIR/pert_type_csvs/, then averages the 977 landmark-gene
expression columns per cell_id and writes the result to
$CONTEXTPERT_DATA_DIR/ctrls.csv.
"""
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.getenv("CONTEXTPERT_DATA_DIR", "data"))
PERT_TYPE_DIR = DATA_DIR / "pert_type_csvs"
OUT_FILE = DATA_DIR / "ctrls.csv"

CTL_FILES = ["ctl_untrt.csv", "ctl_vector.csv", "ctl_vehicle.csv"]
NON_GENE_COLS = {
    "inst_id", "cell_id", "pert_id", "pert_type",
    "pert_dose", "pert_dose_unit", "pert_time",
    "sig_id", "distil_cc_q75", "pct_self_rank_q25",
}


def main() -> None:
    frames = []
    for name in CTL_FILES:
        path = PERT_TYPE_DIR / name
        print(f"Loading {path}")
        frames.append(pd.read_csv(path, low_memory=False))
    combined = pd.concat(frames, ignore_index=True)

    gene_cols = [c for c in combined.columns if c not in NON_GENE_COLS]
    print(f"Averaging {len(gene_cols)} gene columns across {combined['cell_id'].nunique()} cell lines")

    ctrls = combined.groupby("cell_id")[gene_cols].mean().reset_index()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ctrls.to_csv(OUT_FILE, index=False)
    print(f"Wrote {OUT_FILE} ({ctrls.shape[0]} rows x {ctrls.shape[1]} cols)")


if __name__ == "__main__":
    main()
