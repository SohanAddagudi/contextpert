"""
Bundled DDR-Bench and DTR-Bench reference data.
"""
import os
from importlib.resources import files

import pandas as pd

__all__ = [
    "ddr_smiles",
    "dtr_smiles",
    "dtr_targets",
    "ref_path",
]


_LINCS_FILES = {
    ("ddr", "lincs"): ("opentargets", "disease_drug_triples_csv", "disease_drug_triples_lincs.csv"),
    ("dtr", "lincs"): ("opentargets", "drug_target_pairs_csv", "drug_target_pairs_lincs.csv"),
}
_FULL_RELATIVES = {
    ("ddr", "full"): ("opentargets", "disease_drug_triples_csv", "disease_drug_triples.csv"),
    ("dtr", "full"): ("opentargets", "drug_target_pairs_csv", "drug_target_pairs.csv"),
}


def ref_path(kind, mode="lincs"):
    """Path to a DDR/DTR-Bench reference CSV."""
    if (kind, mode) in _LINCS_FILES:
        parts = _LINCS_FILES[(kind, mode)]
        return str(files("contextpert").joinpath("data", *parts))
    if (kind, mode) in _FULL_RELATIVES:
        data_dir = os.environ.get("CONTEXTPERT_DATA_DIR")
        if not data_dir:
            raise EnvironmentError(
                f"mode='full' requires the CONTEXTPERT_DATA_DIR environment variable "
                f"to point at the full data release (LINCS-overlap files for {kind.upper()}-Bench "
                f"are bundled with the package and accessible via mode='lincs')."
            )
        return os.path.join(data_dir, *_FULL_RELATIVES[(kind, mode)])
    raise ValueError(f"Unknown (kind, mode)=({kind!r}, {mode!r}). "
                     f"kind must be 'ddr' or 'dtr'; mode must be 'lincs' or 'full'.")


def ddr_smiles(mode="lincs"):
    """Unique SMILES to embed for DDR-Bench."""
    return pd.read_csv(ref_path("ddr", mode))["smiles"].unique().tolist()


def dtr_smiles(mode="lincs"):
    """Unique SMILES to embed for DTR-Bench (drug side)."""
    return pd.read_csv(ref_path("dtr", mode))["smiles"].unique().tolist()


def dtr_targets(mode="lincs"):
    """Unique Ensembl gene IDs to embed for DTR-Bench (target side)."""
    return pd.read_csv(ref_path("dtr", mode))["targetId"].unique().tolist()
