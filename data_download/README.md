# Data Download and Preprocessing Instructions

Instructions for recreating the benchmark datasets from scratch for running the experiments in the main codebase.

## LINCS L1000 Phase I Data
The raw data can be found at the Gene Expression Omnibus (GEO) under accession number **GSE92742**.

### Prerequisites
Set up a directory for the raw data files.
```bash
mkdir data_raw
export CONTEXTPERT_RAW_DATA_DIR=data_raw
```

Download and unzip LINCS L1000 Phase I data to the raw data directory.
```bash
bash 01_download_lincs.sh
```

If not already set, set your processed data directory:
```bash
mkdir data
export CONTEXTPERT_DATA_DIR=data
```

### Run

Filter and clean the data, and add it to the processed data directory by running
```bash
python 02_data_process.py
```
This will
- Read in gctx data file in a memory efficient manner as a pandas dataframe
- Filter to only the landmark 977 genes instead of the full imputed transcriptome.
- Concatenate perturbation/experiment information (perturbation, dose, time, quality filters, etc)
- Save as `lincs_full.csv` in the processed data directory

## Open Targets Cancer Phase IV Drug Data ETL

This directory contains scripts and data for extracting cancer-related phase IV small molecule drugs from Open Targets data.

### Overview

The `etl_ot_cancer_phase4.py` script processes Open Targets data to create curated datasets for cancer drug repurposing analysis. It extracts phase IV small molecule drugs that target proteins associated with cancer diseases.

### Prerequisites

1. **PySpark**: Install Apache Spark and PySpark
   ```bash
   pip install pyspark
   ```

2. **Open Targets Data**: Download Open Targets release data (version 25.06)
   ```bash
   bash 03_download_opentargets.sh
   ```

### Data Processing Pipeline

The script performs the following steps:

#### 1. Data Loading
- Loads disease ontology data from `{CONTEXTPERT_RAW_DATA_DIR}/opentargets/25.06/disease`
- Loads target-disease associations from `{CONTEXTPERT_RAW_DATA_DIR}/opentargets/25.06/association_by_datasource_direct`
- Loads known drug data from `{CONTEXTPERT_RAW_DATA_DIR}/opentargets/25.06/known_drug`

#### 2. Cancer Disease Identification
- Uses therapeutic area `MONDO_0045024` ("cancer or benign tumor") as the cancer root
- Filters diseases that belong to the cancer therapeutic area

#### 3. Phase IV Small Molecule Filtering
- Identifies small molecule drug types from the known_drug dataset
- Filters for phase IV drugs only
- Restricts to small molecule compounds

#### 4. Data Integration
- Creates cancer disease-target pairs from association data
- Generates cancer-target-drug triples by joining filtered data
- Filters to only include diseases with multiple drug targets (for repurposing potential)

### Running the Script

```bash
python 04_etl_ot_cancer_phase4.py
```

### Output Files

The script generates the following files in `{CONTEXTPERT_DATA_DIR}/opentargets`:

#### CSV Formats
- `cancer_disease_target_csv/`: Disease-target pairs for cancer diseases
- `phase4_small_molecule_csv/`: All phase IV small molecule drugs
- `cancer_target_drug_phase4_csv/`: Final cancer-target-drug triples

#### Parquet Formats
- `cancer_disease_target_parquet/`: Disease-target pairs (Parquet)
- `phase4_small_molecule_parquet/`: Phase IV drugs (Parquet)
- `cancer_target_drug_phase4_parquet/`: Final triples (Parquet)

### Key Filtering Criteria

1. **Diseases**: Must be classified under cancer therapeutic area (`MONDO_0045024`)
2. **Drugs**: Must be phase IV and small molecule compounds
3. **Multi-target constraint**: Only diseases with multiple drug targets are included to enable repurposing analysis

### System Requirements

- Memory: 16GB+ (configured in Spark driver)
- Cores: Uses all available local cores (`local[*]`)
- Storage: Sufficient space for intermediate processing and output files

### Data Schema

#### Final Triples Schema:
- `diseaseId`: Open Targets disease identifier
- `diseaseName`: Human-readable disease name
- `targetId`: Open Targets target identifier
- `drugId`: Open Targets drug identifier
- `drugName`: Drug name (preferred name)

### Reproducibility

To reproduce the exact same results:
1. Use Open Targets release 25.06
2. Ensure the same Spark configuration (16GB driver memory)
3. Use the same filtering criteria defined in the script
4. Run on the same input data files

The script includes deterministic ordering (`orderBy("diseaseId")`) to ensure consistent output across runs.

> Once these steps are complete, you will have the necessary benchmarks for running the experiments in the main codebase.

## Disease Mapping: BRD to ChEMBL ID Mapping and Data Merging

This repository contains two Python scripts for mapping BRD (Broad Institute) compound identifiers to ChEMBL IDs and merging the results with additional datasets.

### Files

1. `06_brd_to_chembl.py`: Maps BRD compound identifiers to ChEMBL IDs using the ChEMBL web service API.

**Dependencies:**
```bash
pip install pandas numpy chembl_webresource_client
```

2. `07_chembl_lincs_merge.py`: Merges datasets using ChEMBL IDs as the common identifier. 

**Input Files:**
- Cancer target drug data (following the instructions of the OpenTargets dataset processing pipeline)
- LINCS data with ChEMBL mappings (generated by `06_brd_to_chembl.py`)

## Get FM Embeddings

Finally, we preprocess the expression data using AIDO.Cell to get gene and cell embeddings.

### Installation
Must be run on an ampere (A100) or higher GPU for flash attention support.
```bash
git clone https://github.com/genbio-ai/ModelGenerator.git
cd ModelGenerator
pip install -e ".[flash-attn]"
cd experiments/AIDO.Cell
pip install -r requirements.txt
```

### AIDO.Cell

Get individual gene embeddings and cell-level average embeddings using AIDO.Cell.
We use cell-level averages to represent cell line contexts, and gene-level embeddings to represent gene or target contexts.

```bash
python 99_aido_cell_embedding.py --input $CONTEXTPERT_DATA_DIR/ctrls_symbols.csv --output_base $CONTEXTPERT_DATA_DIR/aido_cell_embeddings --model aido_cell_3m
python 99_aido_cell_embedding.py --input $CONTEXTPERT_DATA_DIR/ctrls_symbols.csv --output_base $CONTEXTPERT_DATA_DIR/aido_cell_embeddings --model aido_cell_10m
python 99_aido_cell_embedding.py --input $CONTEXTPERT_DATA_DIR/ctrls_symbols.csv --output_base $CONTEXTPERT_DATA_DIR/aido_cell_embeddings --model aido_cell_100m
```

> Note: Input CSV's first column must be an ID, and the remaining columns must be gene symbols.
