# Cell-Level Virtual Screening

Ellington, Caleb N., Sohan Addagudi, Jiaqi Wang, Benjamin J. Lengerich, and Eric P. Xing. 2026. “Cell-Level Virtual Screening.” Preprint, bioRxiv, May 13. https://doi.org/10.64898/2026.05.11.724149.

```
@misc{ellington_cell-level_2026,
	title = {Cell-{Level} {Virtual} {Screening}},
	url = {https://www.biorxiv.org/content/10.64898/2026.05.11.724149v1},
	doi = {10.64898/2026.05.11.724149},
	author = {Ellington, Caleb N. and Addagudi, Sohan and Wang, Jiaqi and Lengerich, Benjamin J. and Xing, Eric P.},
	publisher = {bioRxiv},
	month = may,
	year = {2026},
}
```

This repo contains the code to reproduce all datasets, models, and evaluations presented in the above manuscript.

## Interactive Results

[DTR-Bench: Drug-Gene Perturbation Visualization](https://sohanaddagudi.github.io/contextpert/dual_visualization.html)

## Quickstart

Both evals score your representation against OpenTargets–LINCS ground truth. Place the following two files under `$CONTEXTPERT_DATA_DIR/opentargets/`:

- `disease_drug_triples_csv/disease_drug_triples_lincs.csv` — DDR-Bench labels (`smiles`, `targets`, `diseaseId`)
- `drug_target_pairs_csv/drug_target_pairs_lincs.csv` — DTR-Bench labels (`smiles`, `targetId`)

**1. Clone and install**

```bash
git clone --recurse-submodules https://github.com/SohanAddagudi/contextpert.git
cd contextpert
pip install -e Contextualized
pip install -e .

mkdir -p data && export CONTEXTPERT_DATA_DIR=data
rclone bisync box:/Contextualized\ Perturbation\ Modeling $CONTEXTPERT_DATA_DIR -v
```

**2. Produce embeddings on the DDR-Bench molecule list**

```python
import os, pandas as pd
ref = pd.read_csv(os.path.join(os.environ['CONTEXTPERT_DATA_DIR'],
                               'opentargets/disease_drug_triples_csv/disease_drug_triples_lincs.csv'))
smiles_list = ref['smiles'].unique()
my_drug_df = my_model.embed(smiles_list)   # DataFrame: 'smiles' + one column per embedding dim
```

See `example_submissions/sm_cohesion_*_submission.py` for working templates (Morgan, expression, metagenes, AIDO Cell 3M, contextualized networks, random).

**3. Run DDR-Bench**

```python
from contextpert import submit_drug_disease_cohesion
results = submit_drug_disease_cohesion(my_drug_df, mode='lincs')   # prints Hits@k, MRR@k
```

**4. Produce embeddings on the DTR-Bench molecules and targets**

```python
ref = pd.read_csv(os.path.join(os.environ['CONTEXTPERT_DATA_DIR'],
                               'opentargets/drug_target_pairs_csv/drug_target_pairs_lincs.csv'))
my_drug_df   = my_model.embed_drugs(ref['smiles'].unique())      # DataFrame: 'smiles' + embedding cols
my_target_df = my_model.embed_targets(ref['targetId'].unique())  # DataFrame: 'targetId' + embedding cols
```

See `example_submissions/drug_target_*_submission.py` for working templates.

**5. Run DTR-Bench**

```python
from contextpert import submit_drug_target_mapping
results = submit_drug_target_mapping(my_drug_df, my_target_df, mode='lincs')  # prints AUROC, AUPRC, Hits@k
```

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

### Gene Embeddings

#### Gene Embeddings

Pretrained gene embeddings used in **CellVS-Net** can be downloaded from Zenodo and moved into /data/gene_embeddings/:

```bash
curl -L "https://zenodo.org/records/20240447/files/gene_embeddings.zip?download=1" -o gene_embeddings.zip
unzip gene_embeddings.zip
```

These embeddings include multiple pretrained representations of genes across modalities:

#### AIDOcell_100M_Norman_Aligned (D=640)
Cell-contextualized gene embeddings trained on large-scale perturbation data.

#### AIDOdna (D=4352)
DNA sequence-based gene embeddings.

#### chemberta_embeddings.npz
Chemical representation embeddings derived from SMILES-based transformer models.

#### AIDOprot_seq+struct (D=1024)
Protein sequence + structure-aware embeddings.

#### AIDOprot_mean (D=384)
Mean-pooled protein embeddings.

#### PCA_gene_embeddings.h5ad
PCA-reduced gene expression embedding baseline.

## Baseline Representations

### Training Ridge Regression Predictors

`predictors/train_predictors.py` trains simple ridge regression baselines that map drug structure to cellular representations, and aggregates per-gene representations for shRNA perturbations. These are used as baselines for the DDR-Bench and DTR-Bench evaluations (Tables 4 and 5).

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

SPRINT ([panspecies-dti](https://github.com/abhinadduri/panspecies-dti)) generates drug and target embeddings used as a baseline for DDR-Bench and DTR-Bench. The pipeline lives in `sprint/` and runs in four stages.

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

`sprint/01_prepare_inputs.py` builds `drugs.csv` (SMILES from DDR-Bench + DTR-Bench) and `targets.csv` (Ensembl → UniProt → protein sequence) under `$CONTEXTPERT_DATA_DIR/sprint/`.

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

### Table 1 (Pairwise regression MSE on context-held-out split, per perturbation type)

Per-modality MSE for chemical, shRNA, over-expression, and ligand perturbations on a **context-held-out** split, comparing CellVS-Net Target and CellVS-Net Molecule against a population baseline.

The following scripts correspond to the different context-representation settings used in Table 1:

- `table3_cellvsnet_molecule_chemberta.py`
  Uses **ChemBERTa molecular embeddings** as the perturbation context (CellVS-Net Molecule).

- `table3_cellvsnet_molecule_fingerprint.py`
  Uses **Morgan fingerprint representations** as the perturbation context.

- `table_3_cellvsnet_gene.py`
  Uses **target-based context representations** (CellVS-Net Target).

#### Running Table 1 Experiments

The `table_3_cellvsnet_gene.py` script should be run **for all perturbation types** by setting `pert_to_fit_on` to one of:

- `trt_cp` – chemical perturbations
- `trt_sh` – shRNA perturbations
- `trt_oe` – overexpression perturbations
- `trt_lig` – ligand perturbations

`table_3_cellvsnet_gene.py` is preset with the target representations used in Table 1.

### Table 2 (CellVS-Net joint vs separate training across modalities)

Network prediction MSE comparing per-modality (separate) CellVS-Net training against a single joint encoder trained on the union of all four perturbation modalities (chemical, shRNA, over-expression, ligand). Scripts live in `joint_training/`:

```bash
python joint_training/joint_train.py            # train the joint CellVS-Net encoder
python joint_training/sm_cohesion_joint.py      # evaluate joint model on DDR-Bench
python joint_training/drug_target_joint.py      # evaluate joint model on DTR-Bench
```

### Table 3 (DDR-Bench: Disease Retrieval — Predicting Disease Indications for Drugs with Novel Targets)

DDR-Bench evaluates **disease retrieval performance** for small-molecule drug representations: whether virtual screening approaches can capture similarity between drugs that produce similar **cellular effects**, even when they act on **different molecular targets**.

Evaluation is by **Hits@k** with *k* ∈ {1, 5, 10, 25}.

#### Running Table 3 Experiments

This script runs and performs bootstraps + paired significance testing for all representations:

```bash
python table_generation/sm_cohesion_bootstrap.py
```

### Table 4 (DDR-Bench paired bootstrap p-values vs random for Hits@5)

Produced by the same `table_generation/sm_cohesion_bootstrap.py` script as Table 3. Reports one-sided paired bootstrap p-values (10,000 resamples) for Hits@5 versus the random baseline, alongside the all-pairs significance matrix.

### Figure 4 (Drug-organization clustermaps, PC3 cell line)

Generates **drug similarity clustermaps** comparing how drugs organize based on different representations (gene networks, expression, metagenes, or molecular fingerprints) for the PC3 cell type. Drugs are annotated with their FDA-approved disease indications.

```bash
python table_generation/figure2.py --representation metagenes
python table_generation/figure2.py --representation expression
python table_generation/figure2.py --representation morgan
python table_generation/figure2.py --representation network --network_results_dir /path/to/outputs
```

Appendix Figures 5 and 6 extend this analysis to additional cell lines (NPC, MCF7, HT29, HEPG2, A375, A549, HA1E, HCC515) using the same `figure2.py` script with different cell-type inputs.

### Table 5 (DTR-Bench: Drug-Target Retrieval — Matching Synonymous Perturbations Across Modalities)

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

### Table 6 (DTR-Bench summary statistics)

Global summary statistics for the DTR benchmark — dataset size and composition across drugs, targets, perturbations, and evaluation pairs.

- `table_generation/table7_DTR-Bench.py` *(filename retains `table7_`; produces paper Table 6)*

### Table 7 (DDR-Bench coverage by disease)

Per-disease coverage statistics for DDR-Bench — number of distinct target signatures and drugs associated with each disease. Computed from OpenTargets–LINCS disease–drug–target triples.

- `table_generation/table6_DDR-Bench.py` *(filename retains `table6_`; produces paper Table 7)*

### Appendix tables

Selected appendix material reproduces from existing scripts:

**Table 14 (Sample-held-out control-network MSE, Appendix F.5.1)** — MSE of inferred networks on a **sample-held-out split** using **control perturbation measurements** (`ctl_vehicle`, `ctl_vector`, `ctl_untrt`).

```bash
cd table_generation
python table1_controlnetworks.py population            # single global network
python table1_controlnetworks.py cell_specific         # one network per cell line
python table1_controlnetworks.py contextualized        # contextualized network
python table1_controlnetworks.py contextualized_full   # contextualized + dose & time
python table1_controlnetworks.py aggregate
```

Reported MSEs: Train (Full), Test (Full), Test (`n_c > 3`), Test (`n_c <= 3`).

**Table 15 (Continuous-context perturbed-expression MSE, Appendix F.5.2)** — network inference on **perturbed expression** with one-hot vs continuous (dose/time/cell-type) context encodings.

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

**Tables 8–13 (extended encoder ablations and DDR-/DTR-Bench results with confidence intervals)** — reproducible from the same `sm_cohesion_bootstrap.py`, `drug_target_bootstrap.py`, and `joint_training/` scripts with different argument combinations; see the in-script comments for details.
