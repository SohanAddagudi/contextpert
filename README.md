# Virtual Screening on Cellular Systems

[DTR-Bench: Drug-Gene Perturbation Visualization](https://sohanaddagudi.github.io/contextpert/dual_visualization.html)

Todo: submission function intro and leaderboard tables

## Installation

```bash
git clone --recurse-submodules https://github.com/SohanAddagudi/contextpert.git
cd contextpert/Contextualized
pip install -e .
cd ..
pip install -e .
```

Verify installation by running:
```bash
python test_installation.py
```

which should produce a test loss on a dummy dataset.

## Running Experiments

### Recreating Datasets
Create a data directory to store shared data files.

```bash
mkdir data
export CONTEXTPERT_DATA_DIR=data
```

Set up rclone
```bash
conda install conda-forge::rclone
rclone config
# Follow prompts to set up Box remote
```

Sync this with the remote data repository to push or pull any changes to data, results, or other large files. 
Run this at the start and end of your work session to keep everything up to date.
```bash
rclone bisync box:/Contextualized\ Perturbation\ Modeling $CONTEXTPERT_DATA_DIR -v
```

## Create the Dataset from Scratch
Follow instructions in `data_download/README.md` to prepare the data from original sources.

To download preprocessed data, simply run:
```bash
mkdir data
export CONTEXTPERT_DATA_DIR=data  # or your data path
rclone bisync box:/Contextualized\ Perturbation\ Modeling $CONTEXTPERT_DATA_DIR -v
```

For debugging, consider making a smaller version of the dataset with the first 1000 rows
```bash
head -n 1000 data/full_lincs.csv > data/full_lincs_head.csv
```

## Baseline Representations

### Training Ridge Regression Predictors

`predictors/train_predictors.py` trains simple ridge regression baselines that map drug structure to cellular representations, and aggregates per-gene representations for shRNA perturbations. These are used as baselines for the DR-Bench and DTR-Bench evaluations (Tables 4 and 5).

**For `trt_cp` (compound perturbations):** trains ridge models from Morgan fingerprints (2048-dim) to three targets — gene expression (977-dim), PCA metagenes (50-dim), and AIDO Cell 3M embeddings (128-dim). Models are fit on the train split of `trt_cp_split_map.csv`, then used to predict for **all** drugs with valid SMILES.

**For `trt_sh` (gene perturbations):** mean-aggregates train-split instances per gene; PCA is fit on train genes only, then applied to all genes (train + test).

```bash
export CONTEXTPERT_DATA_DIR=data
python predictors/train_predictors.py
```

Outputs are written to `predictors/outputs/`:

- `cp_pred_expression.csv`, `cp_pred_metagenes.csv`, `cp_pred_aido_embeddings.csv` — predicted drug representations (keyed by `pert_id`, `smiles`)
- `sh_repr_expression.csv`, `sh_repr_metagenes.csv`, `sh_repr_aido_embeddings.csv` — aggregated gene representations (keyed by `targetId`)

### Running SPRINT

SPRINT ([panspecies-dti](https://github.com/abhinadduri/panspecies-dti)) generates drug and target embeddings used as a baseline for DR-Bench and DTR-Bench. The pipeline lives in `sprint/` and runs in four stages.

#### 1. Install SPRINT and download the checkpoint

```bash
conda create -n sprint python=3.10 -y
conda activate sprint
pip install git+https://github.com/abhinadduri/panspecies-dti.git
```

Download `sprint.ckpt` from the [SPRINT checkpoints README](https://github.com/abhinadduri/panspecies-dti/blob/main/checkpoints/README.md) and place it in `<panspecies-dti>/checkpoints/`.

Set the required environment variables:

```bash
export CONTEXTPERT_DATA_DIR=data
export SPRINT_DIR=/path/to/panspecies-dti
```

#### 2. Prepare drug and target inputs

`sprint/01_prepare_inputs.py` builds `drugs.csv` (SMILES from DR-Bench + DTR-Bench) and `targets.csv` (Ensembl → UniProt → protein sequence) under `$CONTEXTPERT_DATA_DIR/sprint/`.

```bash
# Structure-aware mode (default): downloads AlphaFold structures, requires FoldSeek step below
python sprint/01_prepare_inputs.py

# Plain-sequence mode: skips structures, fetches AA sequences from UniProt directly
python sprint/01_prepare_inputs.py --no-structure
```

#### 3. (Structure-aware only) Run FoldSeek to generate SaProt sequences

Requires [FoldSeek](https://github.com/steineggerlab/foldseek) on `PATH`. This step converts the downloaded AlphaFold CIF files into structure-aware sequences and writes the final `targets.csv`.

```bash
bash sprint/02a_run_foldseek.sh
```

Skip this step if you used `--no-structure` in step 2.

#### 4. Generate SPRINT embeddings

```bash
bash sprint/02_run_sprint_embed.sh
```

This produces `drug_embeddings.npy` and `target_embeddings.npy` in `$CONTEXTPERT_DATA_DIR/sprint/`.

## Reproducing Figures and Tables

### Table 1 (MSE of inferred networks on a sample-held-out split for control measurements.)

Run `table1_controlnetworks.py` to reproduce the **Table 1** results: MSE of inferred networks on a **sample-held-out split** using **control perturbation measurements** (`ctl_vehicle`, `ctl_vector`, `ctl_untrt`).

The mode is passed as a **positional argument**. Run each mode separately — each saves its results to `table_generation/table1_results2/<mode>.json`.

```bash
cd table_generation
python table1_controlnetworks.py population            # single global network
python table1_controlnetworks.py cell_specific         # one network per cell line
python table1_controlnetworks.py contextualized        # contextualized network
python table1_controlnetworks.py contextualized_full   # contextualized + dose & time
```

After all four modes have been run, aggregate them into the final table:

```bash
python table1_controlnetworks.py aggregate
```

Reported MSEs: Train (Full), Test (Full), Test (`n_c > 3`), Test (`n_c <= 3`).

### Table 2 (MSE of inferred networks on a sample-held-out split for perturbed expression measurements)

This experiment evaluates **network inference performance on perturbed expression data** using a **sample-held-out split**. Models are trained and tested on the same perturbation contexts, but individual samples are held out to assess generalization at the sample level.

#### Running Table 2 Experiments

The mode is passed via the `--mode` flag. Run each of the five modes separately — each saves its results to `table_generation/results/table2_<mode>.json` (or the directory supplied via `--results-dir`).

```bash
cd table_generation
python table2_post_pert_networks.py --mode population
python table2_post_pert_networks.py --mode cell_specific
python table2_post_pert_networks.py --mode contextualized_onehot
python table2_post_pert_networks.py --mode contextualized_expression
python table2_post_pert_networks.py --mode contextualized_expression_full
```

Mode meanings:

- `population` – single global correlation network
- `cell_specific` – one correlation network per (cell, pert) group
- `contextualized_onehot` – contextualized model with one-hot cell context + PCA pert context
- `contextualized_expression` – contextualized model with PCA-of-control-expression cell context
- `contextualized_expression_full` – `contextualized_expression` plus dose, time, and pert-dose-unit features


### Table 3 (MSE of inferred networks on a context-held-out split for various perturbation types using different context representations)

The following scripts correspond to the different context-representation settings used in Table 3:

- `table3_cellvsnet_molecule_chemberta.py`  
  Uses **ChemBERTa molecular embeddings** as the perturbation context.

- `table3_cellvsnet_molecule_fingerprint.py`  
  Uses **molecular fingerprint representations** as the perturbation context.

- `table_3_cellvsnet_gene.py`  
  Uses **target-based context representations**.

#### Running Table 3 Experiments

The `table_3_cellvsnet_gene.py` script should be run **for all perturbation types** by setting `pert_to_fit_on` to one of:

- `trt_cp` – chemical perturbations  
- `trt_sh` – shRNA perturbations  
- `trt_oe` – overexpression perturbations  
- `trt_lig` – ligand perturbations  

`table_3_cellvsnet_gene.py` is preset with the target representations used in table 3.


### Table 4 (Disease Retrieval: Predicting Disease Indications for Drugs with Novel Targets)

This experiment evaluates **disease retrieval performance** for small-molecule drug representations. The goal is to assess whether virtual screening approaches can capture similarity between drugs that produce similar **cellular effects**, even when they act on **different molecular targets**.

Evaluation is performed using **Hits@k** with  
*k* ∈ {1, 5, 10, 25}.

#### Running Table 4 Experiments

This script will run and perform bootstraps and signifiance testing for all representations. 
```
python sm_cohesion_bootstrap.py
```

### Figure 2 (Disease Cohesion Clustermaps) 

This script generates **drug similarity clustermaps** comparing how drugs organize based on different representations (gene networks, expression, metagenes, or molecular fingerprints). Drugs are annotated with their FDA-approved disease indications to visualize whether therapeutically similar drugs cluster together.

```
python table_generation/figure2.py --representation metagenes
python table_generation/figure2.py --representation expression
python table_generation/figure2.py --representation morgan
python table_generation/figure2.py --representation network --network_results_dir /path/to/outputs
```

### Table 5 (Drug-Target Retrieval: Matching Synonymous Perturbations Across Modalities)

This experiment evaluates **cross-modal retrieval** between **small-molecule drug perturbations** and **genetic target perturbations**, testing whether perturbations with similar cell-level effects can be matched across modalities.

Two retrieval tasks are evaluated:

1. **Drug → Target retrieval**  
   Given a small-molecule drug perturbation, retrieve its corresponding genetic target perturbation.

2. **Target → Drug retrieval**  
   Given a genetic perturbation (shRNA knockdown), retrieve small-molecule drugs that target the same gene.

Evaluation metrics are:
- **AUROC** and **AUPRC** for graph reconstruction  
- **Hits@k** for query-level retrieval with *k* ∈ {1, 5, 10, 50}

Expression-based representations are derived from LINCS L1000 small-molecule and shRNA data. PCA-based baselines apply dimensionality reduction to expression features.

#### Running Table 5 Experiments

This script will run and perform bootstraps and signifiance testing for all representations, with batch correction:

```
python drug_target_bootstrap.py
```

### Figure 3
Caleb

### Table 6 (DR-Bench coverage by disease)
This table reports per-disease coverage statistics for DR-Bench, including the number of unique drugs and molecular targets associated with each disease. Coverage is computed from OpenTargets–LINCS disease–drug–target triples and is intended to characterize benchmark composition rather than model performance.

- `table6_DR-Bench.py`  

### Table 7 (DTR-Bench summary statistics)
This table reports global summary statistics for the Drug–Target Retrieval (DTR) benchmark, including dataset size and composition across drugs, targets, perturbations, and evaluation pairs. These statistics provide an overview of benchmark scale and modality coverage.

- `table7_DTR-Bench.py`
