# Open Targets Cancer Phase IV Drug Data ETL

This directory contains scripts and data for extracting cancer-related phase IV small molecule drugs from Open Targets data.

## Overview

The `etl_ot_cancer_phase4.py` script processes Open Targets data to create curated datasets for cancer drug repurposing analysis. It extracts phase IV small molecule drugs that target proteins associated with cancer diseases.

## Prerequisites

1. **PySpark**: Install Apache Spark and PySpark
   ```bash
   pip install pyspark
   ```

2. **Open Targets Data**: Download Open Targets release data (version 25.06)
   ```bash
   export OT_REL=25.06
   export OT_WORK=$PWD/ot_data
   mkdir -p "$OT_WORK/$OT_REL"
   cd        "$OT_WORK/$OT_REL"

   for ds in disease association_by_datasource_direct known_drug; do
     rsync -rpltvz --delete                     \
           rsync.ebi.ac.uk::pub/databases/opentargets/platform/$OT_REL/output/$ds \
           .
   done
   ```

3. **Environment Setup**:
   ```bash
   export OT_WORK=/path/to/your/open-targets-data
   ```

## Data Processing Pipeline

The script performs the following steps:

### 1. Data Loading
- Loads disease ontology data from `{OT_WORK}/{REL}/disease`
- Loads target-disease associations from `{OT_WORK}/{REL}/association_by_datasource_direct`
- Loads known drug data from `{OT_WORK}/{REL}/known_drug`

### 2. Cancer Disease Identification
- Uses therapeutic area `MONDO_0045024` ("cancer or benign tumor") as the cancer root
- Filters diseases that belong to the cancer therapeutic area

### 3. Phase IV Small Molecule Filtering
- Identifies small molecule drug types from the known_drug dataset
- Filters for phase IV drugs only
- Restricts to small molecule compounds

### 4. Data Integration
- Creates cancer disease-target pairs from association data
- Generates cancer-target-drug triples by joining filtered data
- Filters to only include diseases with multiple drug targets (for repurposing potential)

## Running the Script

```bash
cd data/ot
python etl_ot_cancer_phase4.py
```

## Output Files

The script generates the following files in `{OT_WORK}/{REL}/exports/`:

### CSV Formats
- `cancer_disease_target_csv/`: Disease-target pairs for cancer diseases
- `phase4_small_molecule_csv/`: All phase IV small molecule drugs
- `cancer_target_drug_phase4_csv/`: Final cancer-target-drug triples

### Parquet Formats
- `cancer_disease_target_parquet/`: Disease-target pairs (Parquet)
- `phase4_small_molecule_parquet/`: Phase IV drugs (Parquet)
- `cancer_target_drug_phase4_parquet/`: Final triples (Parquet)

## Key Filtering Criteria

1. **Diseases**: Must be classified under cancer therapeutic area (`MONDO_0045024`)
2. **Drugs**: Must be phase IV and small molecule compounds
3. **Multi-target constraint**: Only diseases with multiple drug targets are included to enable repurposing analysis

## System Requirements

- Memory: 16GB+ (configured in Spark driver)
- Cores: Uses all available local cores (`local[*]`)
- Storage: Sufficient space for intermediate processing and output files

## Data Schema

### Final Triples Schema:
- `diseaseId`: Open Targets disease identifier
- `diseaseName`: Human-readable disease name
- `targetId`: Open Targets target identifier
- `drugId`: Open Targets drug identifier
- `drugName`: Drug name (preferred name)

## Reproducibility

To reproduce the exact same results:
1. Use Open Targets release 25.06
2. Ensure the same Spark configuration (16GB driver memory)
3. Use the same filtering criteria defined in the script
4. Run on the same input data files

The script includes deterministic ordering (`orderBy("diseaseId")`) to ensure consistent output across runs.