"""
High-level API for DDR-Bench and DTR-Bench.
"""
from __future__ import annotations

from typing import Callable, Union

import numpy as np
import pandas as pd

from contextpert.data import ddr_smiles, dtr_smiles, dtr_targets
from contextpert.evaluate.drug_disease_cohesion import submit_drug_disease_cohesion
from contextpert.evaluate.drug_target_mapping import submit_drug_target_mapping

__all__ = ["DDRBench", "DTRBench"]


Embedder = Union[Callable[[str], "np.ndarray"], pd.DataFrame, str]


# --------------------------------------------------------------------------- #
# DDR-Bench: drug → disease retrieval                                          #
# --------------------------------------------------------------------------- #

class DDRBench:
    """Drug-Disease Retrieval benchmark."""

    def __init__(self, mode: str = "lincs"):
        self.mode = mode
        self.smiles = ddr_smiles(mode)

    def evaluate(self, embed: Embedder) -> dict:
        """Score ``embed`` (callable, DataFrame, or built-in name)."""
        drug_df = _coerce_drug_df(embed, self.smiles)
        return submit_drug_disease_cohesion(drug_df, mode=self.mode)


# --------------------------------------------------------------------------- #
# DTR-Bench: drug ↔ target retrieval                                           #
# --------------------------------------------------------------------------- #

class DTRBench:
    """Drug-Target Retrieval benchmark."""

    def __init__(self, mode: str = "lincs"):
        self.mode = mode
        self.smiles = dtr_smiles(mode)
        self.target_ids = dtr_targets(mode)

    def evaluate(self, drug_embed: Embedder, target_embed: Embedder) -> dict:
        """Score ``drug_embed`` and ``target_embed`` (each a callable, DataFrame, or built-in name)."""
        drug_df = _coerce_drug_df(drug_embed, self.smiles)
        target_df = _coerce_target_df(target_embed, self.target_ids)
        return submit_drug_target_mapping(drug_df, target_df, mode=self.mode)


# --------------------------------------------------------------------------- #
# Coercion helpers                                                             #
# --------------------------------------------------------------------------- #

def _coerce_drug_df(embed, smiles_list):
    if isinstance(embed, pd.DataFrame):
        return embed
    if isinstance(embed, str):
        return _builtin_drug(embed, smiles_list)
    if callable(embed):
        return _df_from_callable(embed, smiles_list, key_col="smiles")
    raise TypeError(
        f"Expected a callable, DataFrame, or built-in name; got {type(embed).__name__}"
    )


def _coerce_target_df(embed, target_id_list):
    if isinstance(embed, pd.DataFrame):
        return embed
    if isinstance(embed, str):
        return _builtin_target(embed, target_id_list)
    if callable(embed):
        return _df_from_callable(embed, target_id_list, key_col="targetId")
    raise TypeError(
        f"Expected a callable, DataFrame, or built-in name; got {type(embed).__name__}"
    )


def _df_from_callable(fn, keys, key_col):
    """Apply ``fn`` to each key and return a DataFrame with ``key_col`` + embedding columns."""
    vecs = np.stack([np.asarray(fn(k)).ravel() for k in keys])
    df = pd.DataFrame(vecs, columns=[f"dim_{i}" for i in range(vecs.shape[1])])
    return df.assign(**{key_col: list(keys)})


def _builtin_drug(name, smiles_list):
    name = name.lower()
    if name == "morgan":
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles_list])
        return pd.DataFrame(fps, columns=[f"dim_{i}" for i in range(2048)]).assign(smiles=smiles_list)
    if name == "random":
        rng = np.random.default_rng(1)
        vecs = rng.standard_normal((len(smiles_list), 100)).astype(np.float32)
        return pd.DataFrame(
            vecs, columns=[f"drug_dim_{i}" for i in range(100)]
        ).assign(smiles=smiles_list)
    raise ValueError(
        f"Unknown built-in drug baseline {name!r}. Available: 'morgan', 'random'."
    )


def _builtin_target(name, target_id_list):
    name = name.lower()
    if name == "random":
        rng = np.random.default_rng(1)
        vecs = rng.standard_normal((len(target_id_list), 100)).astype(np.float32)
        return pd.DataFrame(
            vecs, columns=[f"target_dim_{i}" for i in range(100)]
        ).assign(targetId=target_id_list)
    raise ValueError(
        f"Unknown built-in target baseline {name!r}. Available: 'random'."
    )
