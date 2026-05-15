import os
from pathlib import Path
import pandas as pd

# Config
DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])
TRIPLES_LINCS = DATA_DIR / "opentargets" / "disease_drug_triples_csv" / "disease_drug_triples_lincs.csv"
OUTDIR = Path("paper_tables_from_eval")
OUTDIR.mkdir(parents=True, exist_ok=True)

TOP_N = 10


def main():
    print("="*80)
    print("GENERATING TABLE 6: DISEASE-TARGET-DRUG COVERAGE")
    print("="*80)
    
    # Load evaluation triples
    triples = pd.read_csv(TRIPLES_LINCS)
    
    # Verify required columns
    need = {"smiles", "targets", "diseaseId", "diseaseName"}
    missing = need - set(triples.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Print verification statistics
    print(f"\n{len(triples)} drug indications. {triples['smiles'].nunique()} unique drugs, "
          f"{triples['targets'].nunique()} unique target signatures, "
          f"{triples['diseaseId'].nunique()} unique diseases\n")
    
    # Compute per-disease statistics
    per_disease_targets = (
        triples[["diseaseId", "diseaseName", "targets"]]
        .drop_duplicates()
        .groupby(["diseaseId", "diseaseName"])
        .size()
        .rename("n_targets")
    )
    
    per_disease_drugs = (
        triples[["diseaseId", "diseaseName", "smiles"]]
        .drop_duplicates()
        .groupby(["diseaseId", "diseaseName"])
        .size()
        .rename("n_drugs")
    )
    
    # Combine and sort
    table1 = (
        pd.concat([per_disease_targets, per_disease_drugs], axis=1)
        .reset_index()
        .fillna(0)
        .astype({"n_targets": "int64", "n_drugs": "int64"})
        .sort_values(["n_targets", "n_drugs", "diseaseName"], ascending=[False, False, True])
    )
    
    # Save outputs
    full_csv = OUTDIR / "appendix_A1_diseases_full.csv"
    table1.to_csv(full_csv, index=False)
    
    top_csv = OUTDIR / "table1_main_topN.csv"
    table1.head(TOP_N).to_csv(top_csv, index=False)
    
    print(f"Full table:  {full_csv}")
    print(f"Top-{TOP_N} table: {top_csv}")
    print(f"\nTop 5 diseases by target count:")
    for i, (_, row) in enumerate(table1.head(5).iterrows(), 1):
        print(f"  {i}. {row['diseaseName']}: {row['n_targets']} targets, {row['n_drugs']} drugs")


if __name__ == "__main__":
    main()
