#!/usr/bin/env python
import os
from pyspark.sql import SparkSession, functions as F

REL  = "25.06"
ROOT = f"{os.environ['OT_WORK']}/{REL}"
OUT  = f"{ROOT}/exports"

spark = (
    SparkSession.builder
    .appName("ot-cancer-phase4")
    .master("local[*]")
    .config("spark.driver.memory","16g")
    .getOrCreate()
)

dz    = spark.read.parquet(f"{ROOT}/disease")
assoc = spark.read.parquet(f"{ROOT}/association_by_datasource_direct")
kd    = spark.read.parquet(f"{ROOT}/known_drug")

print(f"Loaded - disease {dz.count():,} | assoc {assoc.count():,} | kd {kd.count():,}")

"""
# 1 figure out which ontology node is the cancer root
def get_cancer_root():
    for label in ["cancer or benign tumour",
                  "cancer (disease)",
                  "cancer"]:
        hit = (dz.filter(F.lower("name") == label)
                 .select("id").limit(1).collect())
        if hit:
            return hit[0]["id"]
    return "OTAR_0000015"

CANCER_ROOT = get_cancer_root()
print("Using cancer root =", CANCER_ROOT)


# 2 cancer diseases
cancers = (dz.filter(F.array_contains("ancestors", CANCER_ROOT))
             .select(F.col("id").alias("diseaseId"),
                     F.col("name").alias("diseaseName")))
print(f"Cancer diseases: {cancers.count():,}")
cancers.show(3, truncate=False)   
"""

"""
ta_roots = (
    dz.filter("ontology.isTherapeuticArea")
      .filter(F.lower("name").like("%cancer%"))
      .select("id", "name")
)
ta_roots.show(truncate=False)

dz.selectExpr("explode(therapeuticAreas) as ta") \
  .groupBy("ta").count() \
  .orderBy("count", ascending=False).show(10, False)

CANCER_ROOT = "OTAR_0000015"
cancers = (dz.filter(F.array_contains("therapeuticAreas", CANCER_ROOT))
             .select(F.col("id").alias("diseaseId"),
                     F.col("name").alias("diseaseName")))

print("Cancer diseases:", cancers.count())
cancers.show(3, truncate=False)
"""

# ta_roots = (dz.filter("ontology.isTherapeuticArea")
#               .select("id","name")
#               .orderBy("name"))
# ta_roots.show(truncate=False)

# TA root
CANCER_TA = "MONDO_0045024"        # cancer or benign tumor

cancers = (
    dz.filter(F.array_contains("therapeuticAreas", CANCER_TA))
      .select(F.col("id").alias("diseaseId"),
              F.col("name").alias("diseaseName"))
)
print("Cancer diseases:", cancers.count()) 
cancers.show(5, truncate=False)


# phase-IV small molecules
SMOL_TYPES = [row["drugType"] for row in
              kd.select("drugType").distinct().collect()
              if "small" in (row["drugType"] or "").lower()]

kd_filt = (
    kd.filter((F.col("phase") == 4.0) & F.col("drugType").isin(SMOL_TYPES))
      .select("diseaseId","targetId","drugId","prefName")
)
print(f"Phase-IV small molecules: {kd_filt.count():,}")
kd_filt.show(5, truncate=False)

# cancer (disease,target) pairs
cancer_dz_tar = (assoc.join(cancers, "diseaseId")
                         .select("diseaseId","targetId")
                         .dropDuplicates())
print(f"Cancer disease-target pairs: {cancer_dz_tar.count():,}")
cancer_dz_tar.show(5, truncate=False)

# final triples
triples = (kd_filt.join(cancers, "diseaseId")
                    .select("diseaseId","diseaseName",
                            "targetId","drugId",
                            F.col("prefName").alias("drugName"))
                    .dropDuplicates())
print(f"Final cancer-target-drug triples: {triples.count():,}")
triples.show(10, truncate=False)

# filter for diseases with multiple targets (for drug repurposing)
# count how many unique targets each disease has in the triples
disease_target_counts = (
    triples.select("diseaseId", "targetId")
           .dropDuplicates()
           .groupBy("diseaseId")
           .agg(F.count("targetId").alias("target_count"))
)

# keep only diseases with multiple targets
diseases_with_multiple_targets = (
    disease_target_counts.filter(F.col("target_count") > 1)
                         .select("diseaseId")
)

print(f"Diseases with multiple targets in drug data: {diseases_with_multiple_targets.count():,}")

# filter triples to only include diseases with multiple targets
triples_filtered = (
    triples.join(diseases_with_multiple_targets, "diseaseId")
)

print(f"Filtered cancer-target-drug triples (multi-target diseases only): {triples_filtered.count():,}")
print(f"Unique disease IDs in filtered triples: {triples_filtered.select('diseaseId').distinct().count():,}")
triples_filtered.show(10, truncate=False)

# write outputs
def single_csv(df, path):
    (df.coalesce(1)
       .write.option("header", True)
       .mode("overwrite")
       .csv(path))

os.makedirs(OUT, exist_ok=True)

# create filtered disease-target pairs from filtered triples
cancer_dz_tar_filtered = (
    triples_filtered.select("diseaseId", "targetId")
                    .dropDuplicates()
                    .orderBy("diseaseId")
)

# order the final triples by disease ID for CSV output
triples_filtered_ordered = triples_filtered.orderBy("diseaseId")

single_csv(cancer_dz_tar_filtered, f"{OUT}/cancer_disease_target_csv")
single_csv(kd_filt,       f"{OUT}/phase4_small_molecule_csv")
single_csv(triples_filtered_ordered,       f"{OUT}/cancer_target_drug_phase4_csv")

cancer_dz_tar_filtered.write.parquet(f"{OUT}/cancer_disease_target_parquet", mode="overwrite")
kd_filt.write.parquet(    f"{OUT}/phase4_small_molecule_parquet",   mode="overwrite")
triples_filtered_ordered.write.parquet(    f"{OUT}/cancer_target_drug_phase4_parquet", mode="overwrite")

print("✓ Done - files in", OUT)
spark.stop()
