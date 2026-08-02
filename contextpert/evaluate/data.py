from importlib.resources import files


_DDR_FILES = {
    "lincs": ("opentargets", "disease_drug_triples_csv", "disease_drug_triples_lincs.csv"),
    "full":  ("opentargets", "disease_drug_triples_csv", "disease_drug_triples.csv"),
}

_DTR_FILES = {
    "lincs": ("opentargets", "drug_target_pairs_csv", "drug_target_pairs_lincs.csv"),
    "full":  ("opentargets", "drug_target_pairs_csv", "drug_target_pairs.csv"),
}


def ddr_ref_path(mode):
    """path to the DDR-Bench reference CSV for mode."""
    if mode not in _DDR_FILES:
        raise ValueError(f"Invalid mode: {mode}. Must be 'full' or 'lincs'")
    return str(files("contextpert").joinpath("data", *_DDR_FILES[mode]))


def dtr_ref_path(mode):
    """path to the DTR-Bench reference CSV for mode."""
    if mode not in _DTR_FILES:
        raise ValueError(f"Invalid mode: {mode}. Must be 'full' or 'lincs'")
    return str(files("contextpert").joinpath("data", *_DTR_FILES[mode]))
