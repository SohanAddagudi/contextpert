import os
import sys
import mygene
import pandas as pd
from pathlib import Path
from typing import Dict, List

# ===================================================================
# 1. Configuration & Constants
# ===================================================================
DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')

PERT_INFO_FILE = os.path.join(DATA_DIR, 'lincs', 'GSE92742_Broad_LINCS_pert_info.txt')
FULL_LINCS_WITH_CHEMBL_FILE = Path(DATA_DIR) / 'full_lincs_with_chembl.csv'
DRUG_TARGET_PAIRS_FILE = Path(DATA_DIR) / 'opentargets/drug_target_pairs_csv/drug_target_pairs.csv'


INST_CHEMBL_FILE = Path(DATA_DIR) / 'inst_chembl.csv'
PERT_TARGET_OUTFILE = Path(DATA_DIR) / 'pert_target.csv'

PERT_TYPES_TO_KEEP: List[str] = ['trt_oe', 'trt_lig', 'trt_sh', 'trt_cp']


# ===================================================================
# 2. Pipeline Functions
# ===================================================================

def generate_inst_chembl_mapping(
    infile: Path,
    outfile: Path
) -> None:
    """Creates a mapping file from instance ID to ChEMBL ID."""
    print("--- Part 1: Generating inst_id to ChEMBL ID Mapping ---")
    print(f"Reading full LINCS data with ChEMBL IDs from: {infile}")
    df = pd.read_csv(infile)
    mapping_df = df[['inst_id', 'chembl_id', 'pert_id']].copy()
    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(outfile, index=False)
    print(f"Saved inst_id -> chembl_id map to: {outfile}")

def create_target_symbol_map(
    drug_target_file: Path
) -> pd.DataFrame:
    """Maps Ensembl gene IDs from the drug-target file to HGNC gene symbols."""
    print("\n--- Part 2: Mapping Ensembl IDs to Gene Symbols ---")
    print(f"Loading drug-target pairs from: {drug_target_file}")
    df = pd.read_csv(drug_target_file)

    ensg_ids = df['targetId'].unique().tolist()
    print(f"Found {len(ensg_ids)} unique Ensembl IDs to query.")

    mg = mygene.MyGeneInfo()
    gene_info = mg.querymany(
        ensg_ids, scopes='ensembl.gene', fields='symbol', species='human', as_dataframe=False
    )

    id_to_symbol_map: Dict[str, str] = {
        info['query']: info.get('symbol')
        for info in gene_info if 'symbol' in info
    }

    df['symbol'] = df['targetId'].map(id_to_symbol_map)
    df = df.drop('targetId', axis=1)
    cols = ['drugId', 'symbol', 'smiles', 'prefName']
    symbols_df = df[cols].copy()
    
    print("Successfully created drug-to-gene-symbol mapping.")
    return symbols_df


def map_perturbations_to_targets(
    pert_info_file: str,
    inst_chembl_file: Path,
    symbols_df: pd.DataFrame,
    outfile: Path
) -> None:
    """Maps compound perturbations (trt_cp) to their gene targets via ChEMBL IDs."""
    print("\n--- Part 3: Mapping Perturbations to Gene Targets ---")

    print(f"Loading perturbation info from: {pert_info_file}")
    pertinfo = pd.read_csv(pert_info_file, sep='\t')
    filtered_df = pertinfo[pertinfo['pert_type'].isin(PERT_TYPES_TO_KEEP)]
    final_df = filtered_df[['pert_id', 'pert_iname', 'pert_type']].copy()

    print(f"Loading pert_id to ChEMBL ID map from: {inst_chembl_file}")
    inst_chembl = pd.read_csv(inst_chembl_file)

    pert_to_chembl = dict(zip(inst_chembl['pert_id'], inst_chembl['chembl_id']))
    symbols_df_unique = symbols_df.drop_duplicates(subset='drugId', keep='first')
    chembl_to_symbol = dict(zip(symbols_df_unique['drugId'], symbols_df_unique['symbol']))

    print("Mapping 'trt_cp' perturbations to their gene targets via ChEMBL ID...")
    trt_cp_mask = final_df['pert_type'] == 'trt_cp'
    initial_cp_count = trt_cp_mask.sum()

    final_df.loc[trt_cp_mask, 'pert_iname'] = (
        final_df.loc[trt_cp_mask, 'pert_id']
        .map(pert_to_chembl)
        .map(chembl_to_symbol)
    )

    final_df.dropna(subset=['pert_iname'], inplace=True)
    final_cp_count = (final_df['pert_type'] == 'trt_cp').sum()
    removed_count = initial_cp_count - final_cp_count

    print("\n--- 'trt_cp' Mapping Statistics ---")
    print(f"Initial 'trt_cp' count: {initial_cp_count}")
    print(f"Final 'trt_cp' count after mapping: {final_cp_count}")
    print(f"Removed (no symbol found): {removed_count}")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(outfile, index=False)
    print(f"\nSuccessfully saved final perturbation-target data to: {outfile}")


# ===================================================================
# 3. Main Execution Block
# ===================================================================

if __name__ == "__main__":
    generate_inst_chembl_mapping(
        infile=FULL_LINCS_WITH_CHEMBL_FILE,
        outfile=INST_CHEMBL_FILE
    )

    target_symbols_df = create_target_symbol_map(
        drug_target_file=DRUG_TARGET_PAIRS_FILE
    )

    map_perturbations_to_targets(
        pert_info_file=PERT_INFO_FILE,
        inst_chembl_file=INST_CHEMBL_FILE,
        symbols_df=target_symbols_df,
        outfile=PERT_TARGET_OUTFILE
    )

    print("\nPipeline finished.")
