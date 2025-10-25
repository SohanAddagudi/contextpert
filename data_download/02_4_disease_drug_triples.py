#!/usr/bin/env python
"""
Extract Disease-Drug Triples with Target Signatures

Creates disease-drug associations where each drug has a "target signature" -
a unique, ordered combination of all targets it hits. Filters to ensure each
disease has at least 2 drugs with unique target signatures, guaranteeing valid
leave-one-signature-out evaluation.

Input:
    ${CONTEXTPERT_RAW_DATA_DIR}/opentargets/{REL}/disease
    ${CONTEXTPERT_RAW_DATA_DIR}/opentargets/{REL}/known_drug

Output:
    ${CONTEXTPERT_DATA_DIR}/opentargets/disease_drug_triples_csv/disease_drug_triples.csv
    Columns: diseaseId, drugId, smiles, prefName, targets
"""

import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

from contextpert.utils import chembl_to_smiles_batch

REL = "25.06"
ROOT = os.environ['CONTEXTPERT_RAW_DATA_DIR'] + f"/opentargets/{REL}"
OUT = os.environ['CONTEXTPERT_DATA_DIR'] + f"/opentargets"

print("=" * 80)
print("EXTRACTING DISEASE-DRUG TRIPLES WITH TARGET SIGNATURES")
print("=" * 80)

# Initialize Spark
spark = (
    SparkSession.builder
    .appName("disease-drug-triples")
    .master("local[*]")
    .config("spark.driver.memory", "16g")
    .getOrCreate()
)

# Load data
print(f"\nLoading OpenTargets data from: {ROOT}")
dz = spark.read.parquet(f"{ROOT}/disease")
kd = spark.read.parquet(f"{ROOT}/known_drug")

print(f"Loaded:")
print(f"  Diseases: {dz.count():,}")
print(f"  Known drugs: {kd.count():,}")

# Cancer therapeutic area
CANCER_TA = "MONDO_0045024"  # cancer or benign tumor

# Identify cancer diseases
cancers = (
    dz.filter(F.array_contains("therapeuticAreas", CANCER_TA))
      .select(F.col("id").alias("diseaseId"),
              F.col("name").alias("diseaseName"))
)
print(f"\nCancer diseases: {cancers.count():,}")

# Filter for Phase IV small molecules
SMOL_TYPES = [row["drugType"] for row in
              kd.select("drugType").distinct().collect()
              if "small" in (row["drugType"] or "").lower()]

print(f"\nSmall molecule types: {SMOL_TYPES}")

kd_filt = (
    kd.filter((F.col("phase") == 4.0) & F.col("drugType").isin(SMOL_TYPES))
      .select("diseaseId", "targetId", "drugId", "prefName")
)

print(f"\nPhase IV small molecule records: {kd_filt.count():,}")

# Join with cancer diseases
cancer_drugs = (
    kd_filt.join(cancers, "diseaseId")
           .select("diseaseId", "diseaseName", "drugId", "targetId", "prefName")
)

print(f"\nCancer drug records: {cancer_drugs.count():,}")

# Create target signatures: for each (diseaseId, drugId), collect all targets as sorted array
print(f"\nCreating target signatures...")
drug_targets = (
    cancer_drugs
    .groupBy("diseaseId", "diseaseName", "drugId", "prefName")
    .agg(
        F.collect_set("targetId").alias("targets_set"),  # collect_set ensures uniqueness
    )
    .withColumn("targets", F.array_sort("targets_set"))  # Sort lexicographically for consistent ordering
    .withColumn("targetSignature", F.concat_ws("|", "targets"))  # Create signature string for grouping
    .select("diseaseId", "diseaseName", "drugId", "prefName", "targetSignature", "targets")
)

print(f"\nDisease-drug pairs with target signatures: {drug_targets.count():,}")
print(f"  Unique diseases: {drug_targets.select('diseaseId').distinct().count():,}")
print(f"  Unique drugs: {drug_targets.select('drugId').distinct().count():,}")
print(f"  Unique target signatures: {drug_targets.select('targetSignature').distinct().count():,}")

drug_targets.show(10, truncate=False)

# Filter to diseases with at least 2 unique target signatures
print(f"\nFiltering to diseases with 2+ unique target signatures...")
disease_sig_counts = (
    drug_targets
    .groupBy("diseaseId")
    .agg(F.countDistinct("targetSignature").alias("unique_signatures"))
)

valid_diseases = (
    disease_sig_counts
    .filter(F.col("unique_signatures") >= 2)
    .select("diseaseId")
)

print(f"\nDiseases with 2+ unique target signatures: {valid_diseases.count():,}")

# Filter drug_targets to only valid diseases
triples_filtered = (
    drug_targets
    .join(valid_diseases, "diseaseId")
    .distinct()
    .orderBy("diseaseId", "diseaseName", "drugId")
)

print(f"\nFiltered disease-drug triples: {triples_filtered.count():,}")
print(f"  Unique diseases: {triples_filtered.select('diseaseId').distinct().count():,}")
print(f"  Unique drugs: {triples_filtered.select('drugId').distinct().count():,}")
print(f"  Unique target signatures: {triples_filtered.select('targetSignature').distinct().count():,}")

triples_filtered.show(10, truncate=False)

# Convert to pandas for SMILES annotation
print(f"\nConverting to pandas for SMILES annotation...")
df_pandas = triples_filtered.toPandas()

# Get unique ChEMBL IDs
unique_chembl_ids = df_pandas['drugId'].unique()
print(f"\n{len(unique_chembl_ids):,} unique ChEMBL IDs to convert to SMILES")

# Batch convert ChEMBL IDs to SMILES
print(f"\nConverting ChEMBL IDs to SMILES (batch mode)...")
chembl_to_smiles_map = chembl_to_smiles_batch(
    unique_chembl_ids,
    batch_size=50,
    show_progress=True
)

# Count successful conversions
successful_conversions = sum(1 for v in chembl_to_smiles_map.values() if v is not None)
success_rate = (successful_conversions / len(unique_chembl_ids)) * 100

print(f"\nConversion results:")
print(f"  Total ChEMBL IDs: {len(unique_chembl_ids):,}")
print(f"  Successfully converted: {successful_conversions:,}")
print(f"  Failed conversions: {len(unique_chembl_ids) - successful_conversions:,}")
print(f"  Success rate: {success_rate:.1f}%")

# Annotate dataframe with SMILES
print(f"\nAnnotating triples with SMILES...")
df_pandas['smiles'] = df_pandas['drugId'].map(chembl_to_smiles_map)

# Count before filtering
total_records = len(df_pandas)
records_with_smiles = df_pandas['smiles'].notna().sum()
records_without_smiles = total_records - records_with_smiles

print(f"\nAnnotation statistics:")
print(f"  Total records: {total_records:,}")
print(f"  Records with SMILES: {records_with_smiles:,}")
print(f"  Records without SMILES: {records_without_smiles:,}")

# Drop rows without SMILES
df_pandas = df_pandas[df_pandas['smiles'].notna()].copy()

# Convert targets list column to string for CSV output
df_pandas['targets'] = df_pandas['targets'].apply(lambda x: '|'.join(x))

# Reorder columns - keep only targets (drop targetSignature as it's redundant)
df_pandas = df_pandas[['diseaseId', 'diseaseName', 'drugId', 'smiles', 'prefName', 'targets']]

print(f"\nFinal disease-drug triples with SMILES: {len(df_pandas):,}")
print(f"  Unique diseases: {df_pandas['diseaseId'].nunique():,}")
print(f"  Unique drugs: {df_pandas['drugId'].nunique():,}")
print(f"  Unique target signatures: {df_pandas['targets'].nunique():,}")

# Verify all diseases still have 2+ unique signatures after SMILES filtering
disease_sig_check = df_pandas.groupby('diseaseId')['targets'].nunique()
invalid_diseases = disease_sig_check[disease_sig_check < 2]
if len(invalid_diseases) > 0:
    print(f"\nWARNING: {len(invalid_diseases)} diseases now have < 2 unique signatures after SMILES filtering")
    print("Removing these diseases...")
    df_pandas = df_pandas[~df_pandas['diseaseId'].isin(invalid_diseases.index)].copy()
    print(f"Final count after removing invalid diseases: {len(df_pandas):,}")
    print(f"  Unique diseases: {df_pandas['diseaseId'].nunique():,}")

# Sample of annotated data
print(f"\nSample of annotated data:")
print(df_pandas.head(10).to_string())

# Write output (CSV only)
output_dir = f"{OUT}/disease_drug_triples_csv"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/disease_drug_triples.csv"

print(f"\nWriting disease-drug triples to: {output_file}")
df_pandas.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
print(f"\nOutput: {output_file}")
print(f"  Total triples with SMILES: {len(df_pandas):,}")
print(f"  Diseases: {df_pandas['diseaseId'].nunique():,}")
print(f"  Drugs: {df_pandas['drugId'].nunique():,}")
print(f"  Unique target signatures: {df_pandas['targets'].nunique():,}")
print("\n✓ Done")

spark.stop()
