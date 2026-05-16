## Preprocessing Instructions

This project uses gene expression data from the **LINCS L1000 Phase I** collection. The following steps outline how to download and preprocess the required data.

### Expression Data

The raw data can be found at the Gene Expression Omnibus (GEO) under accession number **GSE92742**.

1.  **Download the Data**: Navigate to the GEO accession page:
    - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742
    - From the download section at the bottom of the page,
        - download the **Level 3** data file: `GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz`
        - download the landmark gene info: `GSE92742_Broad_LINCS_gene_info_delta_landmark.txt.gz`
        - download the perturbation info (contains smiles): `GSE92742_Broad_LINCS_pert_info.txt.gz`
        - download the inst info: `GSE92742_Broad_LINCS_inst_info.txt.gz`
        - download the sig metrics: `GSE92742_Broad_LINCS_sig_metrics.txt.gz`
        - make sure to 'gunzip' all the files to decompress

2.  **Filter/Clean the Data**: Run the code in the `data_process.ipynb` file. This code will:
    - Read in gctx data file in a memory efficient manner as a pandas dataframe
    - Filter to only the landmark 977 genes instead of the full imputed transcriptome.
    - Concatenate perturbation/experiment information (dose, time, quality filters, etc)
    - Save as csv

### Gene Embeddings

## Gene Embeddings

Pretrained gene embeddings used in **CellVS-Net** can be downloaded from Zenodo and moved into /data/gene_embeddings/:

```bash
curl -L "https://zenodo.org/records/20240447/files/gene_embeddings.zip?download=1" -o gene_embeddings.zip
unzip gene_embeddings.zip
```

These embeddings include multiple pretrained representations of genes across modalities:

## AIDOcell_100M_Norman_Aligned (D=640)
Cell-contextualized gene embeddings trained on large-scale perturbation data.

## AIDOdna (D=4352)
DNA sequence-based gene embeddings.

## chemberta_embeddings.npz
Chemical representation embeddings derived from SMILES-based transformer models.

## AIDOprot_seq+struct (D=1024)
Protein sequence + structure-aware embeddings.

## AIDOprot_mean (D=384)
Mean-pooled protein embeddings.

## PCA_gene_embeddings.h5ad
PCA-reduced gene expression embedding baseline.
