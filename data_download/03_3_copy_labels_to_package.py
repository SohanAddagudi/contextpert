#!/usr/bin/env python
"""
Copy DDR/DTR-Bench Labels into Package Source

Copies the four label CSVs from `${CONTEXTPERT_DATA_DIR}/opentargets/` into
the `contextpert` package source tree at `<repo>/contextpert/data/opentargets/`.
The package only ever reads these bundled files at runtime, so end users do
not need to set CONTEXTPERT_DATA_DIR to evaluate against the benchmarks. Run
this whenever the upstream labels are regenerated.

Input:
    ${CONTEXTPERT_DATA_DIR}/opentargets/disease_drug_triples_csv/disease_drug_triples.csv
    ${CONTEXTPERT_DATA_DIR}/opentargets/disease_drug_triples_csv/disease_drug_triples_lincs.csv
    ${CONTEXTPERT_DATA_DIR}/opentargets/drug_target_pairs_csv/drug_target_pairs.csv
    ${CONTEXTPERT_DATA_DIR}/opentargets/drug_target_pairs_csv/drug_target_pairs_lincs.csv

Output:
    <repo>/contextpert/data/opentargets/disease_drug_triples_csv/disease_drug_triples.csv
    <repo>/contextpert/data/opentargets/disease_drug_triples_csv/disease_drug_triples_lincs.csv
    <repo>/contextpert/data/opentargets/drug_target_pairs_csv/drug_target_pairs.csv
    <repo>/contextpert/data/opentargets/drug_target_pairs_csv/drug_target_pairs_lincs.csv
"""

import os
import shutil
from pathlib import Path

DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')
PKG_DATA_DIR = Path(__file__).resolve().parent.parent / 'contextpert' / 'data'

print("=" * 80)
print("COPYING DDR/DTR-BENCH LABELS INTO PACKAGE SOURCE")
print("=" * 80)

label_files = [
    ('opentargets', 'disease_drug_triples_csv', 'disease_drug_triples.csv'),
    ('opentargets', 'disease_drug_triples_csv', 'disease_drug_triples_lincs.csv'),
    ('opentargets', 'drug_target_pairs_csv',    'drug_target_pairs.csv'),
    ('opentargets', 'drug_target_pairs_csv',    'drug_target_pairs_lincs.csv'),
]

print(f"\nSource: {DATA_DIR}")
print(f"Target: {PKG_DATA_DIR}")
print(f"\nFiles to copy: {len(label_files)}")

for parts in label_files:
    src = os.path.join(DATA_DIR, *parts)
    dst = PKG_DATA_DIR.joinpath(*parts)

    if not os.path.exists(src):
        raise FileNotFoundError(f"Expected label file missing: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"\nCopied: {parts[-1]}")
    print(f"  From: {src}")
    print(f"  To:   {dst}")
    print(f"  Size: {dst.stat().st_size:,} bytes")

print("\n" + "=" * 80)
print("LABEL COPY COMPLETE")
print("=" * 80)
print(f"\n{len(label_files)} files copied into {PKG_DATA_DIR}")
print("\n✓ Done")
