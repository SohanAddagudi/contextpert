#!/usr/bin/env python
"""
Extract Disease-Target Pairs from OpenTargets

Reads OpenTargets data and extracts all cancer disease-target associations.

Input:
    ${CONTEXTPERT_RAW_DATA_DIR}/opentargets/{REL}/disease
    ${CONTEXTPERT_RAW_DATA_DIR}/opentargets/{REL}/association_by_datasource_direct

Output:
    ${CONTEXTPERT_DATA_DIR}/opentargets/disease_target_pairs_csv/disease_target_pairs.csv
"""

import os
from pyspark.sql import SparkSession, functions as F

REL = "25.06"
ROOT = os.environ['CONTEXTPERT_RAW_DATA_DIR'] + f"/opentargets/{REL}"
OUT = os.environ['CONTEXTPERT_DATA_DIR'] + f"/opentargets"

print("=" * 80)
print("EXTRACTING DISEASE-TARGET PAIRS")
print("=" * 80)

# Initialize Spark
spark = (
    SparkSession.builder
    .appName("disease-target-pairs")
    .master("local[*]")
    .config("spark.driver.memory", "16g")
    .getOrCreate()
)

# Load data
print(f"\nLoading OpenTargets data from: {ROOT}")
dz = spark.read.parquet(f"{ROOT}/disease")
assoc = spark.read.parquet(f"{ROOT}/association_by_datasource_direct")

print(f"Loaded:")
print(f"  Diseases: {dz.count():,}")
print(f"  Associations: {assoc.count():,}")

# Cancer therapeutic area
CANCER_TA = "MONDO_0045024"  # cancer or benign tumor

# Identify cancer diseases
cancers = (
    dz.filter(F.array_contains("therapeuticAreas", CANCER_TA))
      .select(F.col("id").alias("diseaseId"),
              F.col("name").alias("diseaseName"))
)
print(f"\nCancer diseases: {cancers.count():,}")
cancers.show(5, truncate=False)

# Extract cancer disease-target pairs from associations
cancer_dz_tar = (
    assoc.join(cancers, "diseaseId")
         .select("diseaseId", "diseaseName", "targetId")
         .distinct()  # Ensure deduplication
         .orderBy("diseaseId", "diseaseName", "targetId")
)

print(f"\nCancer disease-target pairs: {cancer_dz_tar.count():,}")
print(f"  Unique diseases: {cancer_dz_tar.select('diseaseId').distinct().count():,}")
print(f"  Unique targets: {cancer_dz_tar.select('targetId').distinct().count():,}")

cancer_dz_tar.show(10, truncate=False)

# Convert to pandas for CSV writing
print(f"\nConverting to pandas for CSV output...")
df_pandas = cancer_dz_tar.toPandas()

# Write output (CSV only)
output_dir = f"{OUT}/disease_target_pairs_csv"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/disease_target_pairs.csv"

print(f"\nWriting disease-target pairs to: {output_file}")
df_pandas.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
print(f"\nOutput: {output_file}")
print(f"  Total pairs: {len(df_pandas):,}")
print("\n✓ Done")

spark.stop()
