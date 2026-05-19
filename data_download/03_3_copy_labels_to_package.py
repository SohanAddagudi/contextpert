#!/usr/bin/env python
"""
Copy DDR/DTR-Bench labels from the local data dir into the package source tree.

Run this as the final step of the data-processing pipeline (after
``03_1_create_lincs_disease_eval.py`` and ``03_2_create_lincs_drug_target_eval.py``).
It refreshes the four CSVs that ship inside ``contextpert/data/opentargets/`` so
the next ``pip install`` of the package picks them up.

The ``contextpert`` Python package only ever reads these bundled files. Users
who only want to evaluate against the benchmarks therefore do not need to
reproduce any of the upstream data-download pipeline; they install the package
and run the evaluators directly. This script is the bridge between the two
worlds — it exists for maintainers regenerating the labels.

Input  : ``${CONTEXTPERT_DATA_DIR}/opentargets/...``
Output : ``<repo>/contextpert/data/opentargets/...``
"""
import os
import shutil
from pathlib import Path

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])
PKG_DATA = Path(__file__).resolve().parent.parent / "contextpert" / "data"

LABEL_FILES = [
    ("opentargets", "disease_drug_triples_csv", "disease_drug_triples_lincs.csv"),
    ("opentargets", "disease_drug_triples_csv", "disease_drug_triples.csv"),
    ("opentargets", "drug_target_pairs_csv",    "drug_target_pairs_lincs.csv"),
    ("opentargets", "drug_target_pairs_csv",    "drug_target_pairs.csv"),
]


def main():
    for parts in LABEL_FILES:
        src = DATA_DIR.joinpath(*parts)
        dst = PKG_DATA.joinpath(*parts)
        if not src.exists():
            raise FileNotFoundError(f"Expected label file missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")


if __name__ == "__main__":
    main()
