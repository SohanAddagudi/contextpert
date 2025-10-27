import os
from pathlib import Path
import pandas as pd
import numpy as np
import mygene

# Config
DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])
DRUG_TARGET_PAIRS = DATA_DIR / "opentargets" / "drug_target_pairs_csv" / "drug_target_pairs_lincs.csv"
TRT_CP_PATH = DATA_DIR / "trt_cp_smiles_qc.csv"
TRT_SH_PATH = DATA_DIR / "trt_sh_qc.csv"
ENTREZ_TO_ENSEMBL_MAP = DATA_DIR / "entrez_to_ensembl_map.csv"
OUTDIR = Path("paper_tables_from_eval")
OUTDIR.mkdir(parents=True, exist_ok=True)

TOP_N = 20


def normalize_ensg(ensg_id):
    """Remove version number from ENSG ID if present"""
    return ensg_id.split('.')[0] if '.' in ensg_id else ensg_id


def map_ensg_to_symbol(ensg_ids):
    """Map ENSG IDs to gene symbols using mygene.info"""
    mg = mygene.MyGeneInfo()
    results = mg.querymany(ensg_ids, scopes='ensembl.gene', fields='symbol',
                          species='human', as_dataframe=False, returnall=True)
    
    ensg_to_symbol = {}
    for item in results['out']:
        ensg_id = item.get('query')
        symbol = item.get('symbol', ensg_id)
        ensg_to_symbol[ensg_id] = symbol
    
    return ensg_to_symbol


def main():
    print("="*80)
    print("GENERATING TABLE 2: DRUG-TARGET MAPPING COVERAGE")
    print("="*80)
    
    # Load drug-target pairs
    pairs_df = pd.read_csv(DRUG_TARGET_PAIRS)
    all_eval_targets = set(pairs_df['targetId'].unique())
    all_eval_targets_normalized = {normalize_ensg(t) for t in all_eval_targets}
    
    # Load LINCS data
    trt_cp_df = pd.read_csv(TRT_CP_PATH)
    bad_smiles = ['-666', 'restricted']
    trt_cp_df = trt_cp_df[~trt_cp_df['canonical_smiles'].isin(bad_smiles)].copy()
    trt_cp_df = trt_cp_df[trt_cp_df['canonical_smiles'].notna()].copy()
    
    trt_sh_df = pd.read_csv(TRT_SH_PATH, low_memory=False)
    metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                     'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75', 'pct_self_rank_q25']
    gene_cols = [col for col in trt_sh_df.columns if col not in metadata_cols]
    
    # Map Entrez to ENSG
    entrez_to_ensembl_df = pd.read_csv(ENTREZ_TO_ENSEMBL_MAP)
    ensembl_col = 'ensembl_id' if 'ensembl_id' in entrez_to_ensembl_df.columns else 'ensembl_gene_id'
    entrez_to_ensembl = dict(zip(
        entrez_to_ensembl_df['entrez_id'].astype(str),
        entrez_to_ensembl_df[ensembl_col]
    ))
    
    # Map gene columns to ENSG
    is_entrez = all(col.isdigit() for col in list(gene_cols)[:5])
    if is_entrez:
        entrez_col_to_ensg = {col: normalize_ensg(entrez_to_ensembl[col]) 
                             for col in gene_cols if col in entrez_to_ensembl}
        all_sh_targets_ensg = set(entrez_col_to_ensg.values())
    else:
        entrez_col_to_ensg = {col: normalize_ensg(col) for col in gene_cols}
        all_sh_targets_ensg = set(entrez_col_to_ensg.values())
    
    # Compute global statistics
    targets_drug_only = all_eval_targets_normalized - all_sh_targets_ensg
    targets_sh_only = all_sh_targets_ensg - all_eval_targets_normalized
    targets_in_overlap = all_eval_targets_normalized & all_sh_targets_ensg
    
    pairs_df['targetId_normalized'] = pairs_df['targetId'].apply(normalize_ensg)
    overlap_pairs = pairs_df[pairs_df['targetId_normalized'].isin(targets_in_overlap)]
    
    print(f"\nGlobal Statistics:")
    print(f"  Targets with drug only: {len(targets_drug_only)}")
    print(f"  Targets with shRNA only: {len(targets_sh_only)}")
    print(f"  Targets in overlap: {len(targets_in_overlap)}")
    print(f"  Unique drugs on overlap: {overlap_pairs['smiles'].nunique()}")
    print(f"  Target-drug pairs on overlap: {len(overlap_pairs)}\n")
    
    # Map ENSG to symbols
    ensg_to_symbol = map_ensg_to_symbol(list(all_eval_targets))
    
    # Infer shRNA targets
    agg_dict_sh = {col: 'mean' for col in gene_cols}
    sh_by_pert = trt_sh_df.groupby('pert_id').agg(agg_dict_sh).reset_index()
    
    pert_to_target_ensg = {}
    for _, row in sh_by_pert.iterrows():
        expr_values = row[gene_cols].values
        target_col = gene_cols[np.argmin(expr_values)]
        if target_col in entrez_col_to_ensg:
            pert_to_target_ensg[row['pert_id']] = entrez_col_to_ensg[target_col]
    
    trt_sh_df['inferred_target_ensg'] = trt_sh_df['pert_id'].map(pert_to_target_ensg)
    
    # Compute per-target statistics
    per_target_stats = []
    for target_ensg in sorted(all_eval_targets):
        target_symbol = ensg_to_symbol.get(target_ensg, target_ensg)
        target_ensg_normalized = normalize_ensg(target_ensg)
        
        target_drugs = set(pairs_df[pairs_df['targetId'] == target_ensg]['smiles'].unique())
        
        # Drug statistics
        drug_samples = trt_cp_df[trt_cp_df['canonical_smiles'].isin(target_drugs)]
        n_drug_samples = len(drug_samples)
        n_unique_drugs = len(target_drugs)
        n_drug_cells = drug_samples['cell_id'].nunique() if n_drug_samples > 0 else 0
        drug_cells = set(drug_samples['cell_id'].unique()) if n_drug_samples > 0 else set()
        
        # shRNA statistics
        if target_ensg_normalized in targets_in_overlap:
            sh_samples = trt_sh_df[trt_sh_df['inferred_target_ensg'] == target_ensg_normalized]
            n_sh_samples = len(sh_samples)
            n_sh_cells = sh_samples['cell_id'].nunique() if n_sh_samples > 0 else 0
            sh_cells = set(sh_samples['cell_id'].unique()) if n_sh_samples > 0 else set()
        else:
            n_sh_samples = n_sh_cells = 0
            sh_cells = set()
        
        per_target_stats.append({
            'target_symbol': target_symbol,
            'drug_samples': n_drug_samples,
            'unique_drugs': n_unique_drugs,
            'drug_cells': n_drug_cells,
            'sh_samples': n_sh_samples,
            'sh_cells': n_sh_cells,
            'cells_both': len(drug_cells & sh_cells)
        })
    
    # Create and sort dataframe
    stats_df = pd.DataFrame(per_target_stats)
    stats_df = stats_df.sort_values(['unique_drugs', 'drug_samples', 'target_symbol'],
                                   ascending=[False, False, True])
    
    # Save outputs
    global_stats = pd.DataFrame([
        {'metric': 'Targets with drug only', 'count': len(targets_drug_only)},
        {'metric': 'Targets with shRNA only', 'count': len(targets_sh_only)},
        {'metric': 'Targets in overlap (drug & shRNA)', 'count': len(targets_in_overlap)},
        {'metric': 'Unique drugs acting on overlap targets', 'count': overlap_pairs['smiles'].nunique()},
        {'metric': 'Target-drug pairs on overlap targets', 'count': len(overlap_pairs)}
    ])
    
    global_csv = OUTDIR / "table2_summary_stats.csv"
    global_stats.to_csv(global_csv, index=False)
    
    full_csv = OUTDIR / "appendix_A2_per_target_full.csv"
    stats_df.to_csv(full_csv, index=False)
    
    top_csv = OUTDIR / "table2_per_target_top20.csv"
    stats_df.head(TOP_N).to_csv(top_csv, index=False)
    
    print(f"Summary stats: {global_csv}")
    print(f"Full table:    {full_csv}")
    print(f"Top-{TOP_N} table:  {top_csv}")
    
    print(f"\nTop 5 targets by unique drugs:")
    for i, (_, row) in enumerate(stats_df.head(5).iterrows(), 1):
        print(f"  {i}. {row['target_symbol']}: {row['unique_drugs']} drugs, {row['drug_samples']} samples")


if __name__ == "__main__":
    main()
