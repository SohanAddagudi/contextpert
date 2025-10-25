#!/usr/bin/env python
"""
Extract Drug-Target Pairs from OpenTargets

Reads OpenTargets known_drug data and extracts all Phase IV small molecule drug-target
associations, annotated with SMILES strings.

Input:
    ${CONTEXTPERT_RAW_DATA_DIR}/opentargets/{REL}/known_drug

Output:
    ${CONTEXTPERT_DATA_DIR}/opentargets/drug_target_pairs_csv/drug_target_pairs.csv
    Columns: drugId, targetId, smiles, prefName
"""

import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

# Add parent directory to path to import chembl_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from chembl_utils import chembl_to_smiles_batch

REL = "25.06"
ROOT = os.environ['CONTEXTPERT_RAW_DATA_DIR'] + f"/opentargets/{REL}"
OUT = os.environ['CONTEXTPERT_DATA_DIR'] + f"/opentargets"

print("=" * 80)
print("EXTRACTING DRUG-TARGET PAIRS")
print("=" * 80)

# Initialize Spark
spark = (
    SparkSession.builder
    .appName("drug-target-pairs")
    .master("local[*]")
    .config("spark.driver.memory", "16g")
    .getOrCreate()
)

# Load known_drug data
print(f"\nLoading OpenTargets known_drug data from: {ROOT}")
kd = spark.read.parquet(f"{ROOT}/known_drug")
print(f"Loaded known_drug: {kd.count():,}")

# Filter for Phase IV small molecules
SMOL_TYPES = [row["drugType"] for row in
              kd.select("drugType").distinct().collect()
              if "small" in (row["drugType"] or "").lower()]

print(f"\nSmall molecule types: {SMOL_TYPES}")

kd_filt = (
    kd.filter((F.col("phase") == 4.0) & F.col("drugType").isin(SMOL_TYPES))
      .select("drugId", "targetId", "prefName")
      .distinct()
      .orderBy("drugId", "targetId")
)

print(f"\nPhase IV small molecule drug-target pairs: {kd_filt.count():,}")
print(f"  Unique drugs: {kd_filt.select('drugId').distinct().count():,}")
print(f"  Unique targets: {kd_filt.select('targetId').distinct().count():,}")

kd_filt.show(10, truncate=False)

# Convert to pandas for SMILES annotation
print(f"\nConverting to pandas for SMILES annotation...")
df_pandas = kd_filt.toPandas()

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
print(f"\nAnnotating drug-target pairs with SMILES...")
df_pandas['smiles'] = df_pandas['drugId'].map(chembl_to_smiles_map)

# Count before filtering
total_records = len(df_pandas)
records_with_smiles = df_pandas['smiles'].notna().sum()
records_without_smiles = total_records - records_with_smiles

print(f"\nAnnotation statistics:")
print(f"  Total records: {total_records:,}")
print(f"  Records with SMILES: {records_with_smiles:,}")
print(f"  Records without SMILES: {records_without_smiles:,}")

# Drop rows without SMILES and reorder columns
df_pandas = df_pandas[df_pandas['smiles'].notna()].copy()
df_pandas = df_pandas[['drugId', 'targetId', 'smiles', 'prefName']]

print(f"\nFinal drug-target pairs with SMILES: {len(df_pandas):,}")
print(f"  Unique drugs: {df_pandas['drugId'].nunique():,}")
print(f"  Unique targets: {df_pandas['targetId'].nunique():,}")

# Sample of annotated data
print(f"\nSample of annotated data:")
print(df_pandas.head(10))

# Write output (CSV only)
output_dir = f"{OUT}/drug_target_pairs_csv"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/drug_target_pairs.csv"

print(f"\nWriting drug-target pairs to: {output_file}")
df_pandas.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
print(f"\nOutput: {output_file}")
print(f"  Total pairs with SMILES: {len(df_pandas):,}")
print(f"  Drugs with SMILES: {df_pandas['drugId'].nunique():,}")
print(f"  Targets: {df_pandas['targetId'].nunique():,}")
print("\n✓ Done")

spark.stop()
