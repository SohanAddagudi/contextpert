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

## Reproducing Figures and Tables

### Table 1 (MSE of inferred networks on a sample-held-out split for control measurements.)
Sohan

### Table 2 (MSE of inferred networks on a sample-held-out split for perturbed expression measurements)
Sohan

### Table 3 (MSE of inferred networks on a context-held-out split for various perturbation types using different context representations)

The following scripts correspond to the different context-representation settings used in Table 3:

- `table3_cellvsnet_molecule_chemberta.py`  
  Uses **ChemBERTa molecular embeddings** as the perturbation context.

- `table3_cellvsnet_molecule_fingerprint.py`  
  Uses **molecular fingerprint representations** as the perturbation context.

- `table_3_cellvsnet_gene.py`  
  Uses **gene-based context representations**.

#### Running Table 3 Experiments

Each script should be run **for all perturbation types** by setting `pert_to_fit_on` to one of:

- `trt_cp` – chemical perturbations  
- `trt_sh` – shRNA perturbations  
- `trt_oe` – overexpression perturbations  
- `trt_lig` – ligand perturbations  

`table_3_cellvsnet_gene.py` is preset with the gene representations used in table 3.


### Table 4 (Disease Retrieval: Predicting Disease Indications for Drugs with Novel Targets)
Caleb

### Figure 1 
Jiaqi

### Table 5 (Drug-Target Retrieval: Matching Synonymous Perturbations Across Modalities)
Caleb

### Figure 2
Caleb

### Table 6 (DR-Bench coverage by disease)
Caleb

### Table 7 (DTR-Bench summary statistics)
Caleb
...

---
Depreicated

Todo: Sohan and Caleb

### Fitting Contextualized Networks

Todo: Sohan

## Run Experiments

### Table 2 Experiments: Post-Perturbation Networks

Use the following scripts depending on the experimental setup:

#### `pert_context.py`

- Uses:
  - One-hot encoded **cell type context**
  - One-hot encoded **perturbation context**
  - Optional inclusion of **dose** and/or **time**
  - can fit on any pert type

#### `cell_ctxt.ipynb`

- Uses:
  - **Embedding-based** or **PCA-compressed** cell type context
- Requires:
  - `ctrls.csv`
  - Embedding `.npy` files 
  - can fit on any pert type

---

### Table 3 Experiments: Generalization to Unseen Perturbations

#### `unseen_pert.py`

- Requires:
  - `trt_cp_smiles.csv` file with only trt_cp perturbations with smiles
  - `ctrls.csv`
  - (Both in BOX)

---
