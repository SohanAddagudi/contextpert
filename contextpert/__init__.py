"""
Benchmarks for cell-level virtual screening.
"""
import os as _os
from importlib.resources import files as _files

_os.environ.setdefault("CONTEXTPERT_DATA_DIR", str(_files("contextpert") / "data"))

from contextpert.benchmarks import DDRBench, DTRBench
from contextpert.evaluate.drug_disease_cohesion import (
    evaluate_drug_disease_cohesion,
    submit_drug_disease_cohesion,
)
from contextpert.evaluate.drug_target_mapping import (
    evaluate_drug_target_mapping,
    submit_drug_target_mapping,
)

__all__ = [
    "DDRBench",
    "DTRBench",
    "submit_drug_disease_cohesion",
    "submit_drug_target_mapping",
    "evaluate_drug_disease_cohesion",
    "evaluate_drug_target_mapping",
]

__all__ = [
    # High-level class API
    "DDRBench",
    "DTRBench",
    # Lower-level functional API
    "submit_drug_disease_cohesion",
    "submit_drug_target_mapping",
    "evaluate_drug_disease_cohesion",
    "evaluate_drug_target_mapping",
]
