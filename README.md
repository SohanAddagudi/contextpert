## Installation

```bash
git clone --recurse-submodules https://github.com/SohanAddagudi/contextpert.git
cd Contextualized
pip install -e .
cd ..
pip install -r requirements.txt
```

Verify installation by running:
```bash
python test_installation.py
```

which should produce a test loss on a dummy dataset.

## Development
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
