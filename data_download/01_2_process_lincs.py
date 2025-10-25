import os
import sys
import h5py
import mygene
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Set, Dict

# ===================================================================
# 1. Configuration & Constants
# ===================================================================
RAW_DATA_DIR = os.getenv('CONTEXTPERT_RAW_DATA_DIR')
DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')


# --- File Paths ---
# Inputs from raw data directory
GCTX_FILE = os.path.join(RAW_DATA_DIR, 'lincs', 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx')
LANDMARK_GENE_FILE = os.path.join(RAW_DATA_DIR, 'lincs', 'GSE92742_Broad_LINCS_gene_info_delta_landmark.txt')
SIG_INFO_FILE = os.path.join(RAW_DATA_DIR, 'lincs', 'GSE92742_Broad_LINCS_sig_info.txt')
SIG_METRICS_FILE = os.path.join(RAW_DATA_DIR, 'lincs', 'GSE92742_Broad_LINCS_sig_metrics.txt')
PERT_INFO_FILE = os.path.join(RAW_DATA_DIR, 'lincs', 'GSE92742_Broad_LINCS_pert_info.txt')

# Outputs to processed data directory
FULL_LINCS_OUTFILE = Path(DATA_DIR) / 'full_lincs.csv'
TRT_CP_FILE = Path(DATA_DIR) / 'trt_cp.csv' 
PROCESSED_CTRLS_OUTFILE = Path(DATA_DIR) / 'ctrls_entrez_avg.csv'
FINAL_SYMBOLS_OUTFILE = Path(DATA_DIR) / 'ctrls_symbols_avg.csv'
TRT_CP_SMILES_OUTFILE = Path(DATA_DIR) / 'trt_cp_smiles.csv' 


# --- Processing Parameters ---
CHUNK_SIZE = 100
CONTROL_CODES: List[str] = [
    "ctl_vehicle", "ctl_vector", "ctl_untrt",
    "ctl_vehicle.cns", "ctl_vector.cns", "ctl_untrt.cns",
    "trt_sh.css"
]
METADATA_NUMERIC_COLS: Set[str] = {
    "pert_dose", "pert_time", "distil_cc_q75", "pct_self_rank_q25"
}

# ===================================================================
# 2. Pipeline Functions
# ===================================================================

def prepare_lincs_data(
    gctx_path: str,
    landmark_gene_path: str,
    sig_info_path: str,
    sig_metrics_path: str,
    output_dir: str,
    full_lincs_output_path: Path
) -> None:
    """
    Processes raw LINCS GCTX data, filters for landmark genes, merges with
    metadata, and saves the full dataset and its subsets by perturbation type.
    """
    print("--- Part 1: Processing Raw LINCS GCTX Data ---")
    
    #-------------------------------------------------------------
    # READ IN EXPRESSION DATA AS PANDAS AND FILTER TO LANDMARK
    #-------------------------------------------------------------

    gene_df = pd.read_csv(landmark_gene_path, sep='\t')
    pr_gene_ids = gene_df['pr_gene_id'].astype(str).tolist()
    data_chunks = []
    
    with h5py.File(gctx_path, 'r') as f:
        data = f['0/DATA/0/matrix']
        row_headers = f['0/META/COL/id'][:].astype(str)
        col_headers = f['0/META/ROW/id'][:].astype(str)
        
        landmark_indices = np.where(np.isin(col_headers, pr_gene_ids))[0]
        if len(landmark_indices) != 978:
            print(f"Warning: Expected 978 landmark genes, but found {len(landmark_indices)}.")

        print(f"Reading expression matrix in chunks of {CHUNK_SIZE}...")
        for i in tqdm(range(0, data.shape[0], CHUNK_SIZE)):
            data_chunk = data[i:i + CHUNK_SIZE, landmark_indices]
            df_chunk = pd.DataFrame(data_chunk, columns=col_headers[landmark_indices], index=row_headers[i:i + CHUNK_SIZE])
            data_chunks.append(df_chunk)

    df = pd.concat(data_chunks)
    df = df.reset_index().rename(columns={'index': 'inst_id'})

    #-------------------------------------------------------------------------
    # ADD PERTURBATION INFO (DOSE, UNIT, QUALITY CONTROLS, PERT_TYPE, etc)
    #-------------------------------------------------------------------------
    print("Merging expression data with perturbation metadata...")
    sig_info = pd.read_csv(sig_info_path, delimiter='\t')
    sig_metrics = pd.read_csv(sig_metrics_path, delimiter='\t')
    info_exploded = sig_info.assign(inst_id=sig_info['distil_id'].str.split('|')).explode('inst_id')
    
    merged_df = pd.merge(df, info_exploded[['cell_id', 'pert_id', 'pert_type', 'pert_dose', 'pert_dose_unit', 'pert_time', 'inst_id', 'sig_id']], on='inst_id', how='left')
    merged_df = merged_df.drop_duplicates(subset='inst_id', keep='first')
    merged_df = pd.merge(merged_df, sig_metrics[['distil_cc_q75', 'pct_self_rank_q25', 'sig_id']], on='sig_id', how='left')
    
    print(f"Saving fully processed LINCS data to: {full_lincs_output_path}")
    merged_df.to_csv(full_lincs_output_path, index=False)

    #-------------------------------------------------------------------------
    # SAVE PERTURBATIONS TYPES AS SEPARATE CSVS
    #-------------------------------------------------------------------------
    print("Saving separate CSVs for each perturbation type...")
    pert_type_col = 'pert_type'
    for pert in merged_df[pert_type_col].unique():
        subset = merged_df[merged_df[pert_type_col] == pert]
        filename = os.path.join(output_dir, f"{pert}.csv")
        subset.to_csv(filename, index=False)
    

def calculate_control_averages(infile: Path, outfile: Path) -> None:
    """
    Reads full LINCS data, filters for control perturbations, and calculates
    the average expression for each cell line.
    """
    print("\n--- Part 2: Calculating Control Averages ---")
    print(f"Reading full LINCS data from: {infile}")
    df = pd.read_csv(infile)

    # Filter for control perturbation types
    df_ctrl = df[df["pert_type"].isin(CONTROL_CODES)].copy()

    # Identify gene expression columns (numeric columns that are Entrez IDs)
    gene_cols = [
        col for col in df_ctrl.select_dtypes(include=np.number).columns
        if col not in METADATA_NUMERIC_COLS
    ]

    # Compute average expression across controls for each cell line
    avg_expr = (
        df_ctrl.groupby("cell_id")[gene_cols]
        .mean()
        .reset_index()
    )

    # Save the processed data
    outfile.parent.mkdir(parents=True, exist_ok=True)
    avg_expr.to_csv(outfile, index=False)


def map_entrez_to_symbols(infile: Path, outfile: Path) -> None:
    """
    Reads a file with Entrez IDs as columns and converts them to
    HGNC gene symbols using the mygene.info API.
    """
    print("\n--- Part 3: Mapping Entrez IDs to Gene Symbols ---")
    print(f"Loading control data from: {infile}")
    df = pd.read_csv(infile)

    # Identify columns that are Entrez IDs (represented as numeric strings)
    entrez_id_cols = [col for col in df.columns if col.isdigit()]
    print(f"Found {len(entrez_id_cols)} Entrez ID columns to convert.")

    # Query mygene.info API
    mg = mygene.MyGeneInfo()
    gene_info = mg.querymany(
        entrez_id_cols,
        scopes="entrezgene",
        fields="symbol",
        species="human",
        as_dataframe=False,
    )

    # Create a mapping from Entrez ID to symbol
    id_to_symbol_map: Dict[str, str] = {
        str(item["query"]): item.get("symbol", item["query"])
        for item in gene_info
        if "symbol" in item
    }

    # Rename DataFrame columns and save
    df.rename(columns=id_to_symbol_map, inplace=True)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)

def add_smiles_to_compounds(
    pert_info_file: str, 
    trt_cp_infile: Path, 
    outfile: Path
) -> None:
    """
    Adds canonical SMILES strings to the compound perturbation data (trt_cp).
    """
    print("\n--- Part 4: Adding SMILES data to Compound Perturbations ---")
    
    # Read perturbation info and filter for compound treatments
    print(f"Loading perturbation info from: {pert_info_file}")
    df_info = pd.read_csv(pert_info_file, sep='\t')
    smiles_df = df_info[df_info['pert_type'] == 'trt_cp']

    # Read the compound expression data generated in Part 1
    print(f"Loading compound expression data from: {trt_cp_infile}")
    pert_df = pd.read_csv(trt_cp_infile, engine='pyarrow')

    # Merge SMILES data into the expression dataframe
    print("Merging SMILES data...")
    smiles_subset = smiles_df[['pert_id', 'canonical_smiles']]
    pert_df = pert_df.merge(smiles_subset, on='pert_id', how='left')

    # Filter out entries with bad SMILES strings and save
    bad_smiles = ['-666', 'restricted']
    pert_df = pert_df[~pert_df['canonical_smiles'].isin(bad_smiles)].reset_index(drop=True)

    print(f"Final shape of compound data with SMILES: {pert_df.shape}")
    pert_df.to_csv(outfile, index=False)


# ===================================================================
# 3. Main Execution Block
# ===================================================================

if __name__ == "__main__":
    # --- Step 1: Process raw GCTX data to get a clean, merged CSV ---
    prepare_lincs_data(
        gctx_path=GCTX_FILE,
        landmark_gene_path=LANDMARK_GENE_FILE,
        sig_info_path=SIG_INFO_FILE,
        sig_metrics_path=SIG_METRICS_FILE,
        output_dir=DATA_DIR,
        full_lincs_output_path=FULL_LINCS_OUTFILE
    )

    # --- Step 2: Use the full CSV to calculate control averages ---
    calculate_control_averages(
        infile=FULL_LINCS_OUTFILE,
        outfile=PROCESSED_CTRLS_OUTFILE
    )

    # --- Step 3: Convert Entrez IDs in the control averages to gene symbols ---
    map_entrez_to_symbols(
        infile=PROCESSED_CTRLS_OUTFILE,
        outfile=FINAL_SYMBOLS_OUTFILE
    )
    
    add_smiles_to_compounds(
        pert_info_file=PERT_INFO_FILE,
        trt_cp_infile=TRT_CP_FILE,
        outfile=TRT_CP_SMILES_OUTFILE
    )

    print("\n Pipeline finished")
