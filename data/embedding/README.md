# Get FM Embeddings

## Installation
```bash
git clone https://github.com/genbio-ai/ModelGenerator.git
cd ModelGenerator
pip install -e ".[flash-attn]"
cd experiments/AIDO.Cell
pip install -r requirements.txt
```

## AIDO.Cell

Get individual gene embeddings and cell-level average embeddings using AIDO.Cell.
We use cell-level averages to represent cell line contexts, and gene-level embeddings to represent gene or target contexts.

```bash
python aido_cell_embedding.py --input <path/to/ctrls_symbols.csv> --output_base <path/to/output_files_prefix> --model <aido_cell_3m/aido_cell_10m/aido_cell_100m>
```

> Note: Input CSV's first column must be an ID, and the remaining columns must be gene symbols.

## Chemberta

This script generates ChemBERTa embeddings for all perturbations (inst_id) using SMILES strings from trt_cp_smiles_qc.csv. It encodes each molecule with the pretrained ChemBERTa model and saves the resulting embeddings

```bash
python chemberta_embeddings.py
```
