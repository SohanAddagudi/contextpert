import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import mygene
import requests
from tqdm import tqdm
import time
from Bio.PDB import alphafold_db
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description='Prepare SPRINT inputs')
parser.add_argument('--no-structure', action='store_true',
                    help='Use plain AA sequences instead of structure-aware sequences')
args = parser.parse_args()

USE_STRUCTURE = not args.no_structure

DATA_DIR = os.environ.get('CONTEXTPERT_DATA_DIR')
if not DATA_DIR:
    raise ValueError("Please set CONTEXTPERT_DATA_DIR environment variable")

OUTPUT_DIR = os.path.join(DATA_DIR, 'sprint')
STRUCTURES_DIR = os.path.join(OUTPUT_DIR, 'alphafold_structures')

# Prepare drugs.csv for SPRINT embedding
def prepare_drugs():
    """
    Prepare drugs.csv for SPRINT embedding.
    """
    print("=" * 80)
    print("PART 1: PREPARING SPRINT DRUG INPUT")
    print("=" * 80)
    
    # Load DTR-Bench ground truth
    dtr_path = os.path.join(DATA_DIR, 'opentargets/drug_target_pairs_csv/drug_target_pairs_lincs.csv')
    print(f"\nLoading DTR-Bench data from: {dtr_path}")
    dtr_df = pd.read_csv(dtr_path)
    print(f"  DTR-Bench pairs: {len(dtr_df):,}")
    print(f"  Unique drugs (SMILES): {dtr_df['smiles'].nunique():,}")
    
    # Load DR-Bench ground truth
    dr_path = os.path.join(DATA_DIR, 'opentargets/disease_drug_triples_csv/disease_drug_triples_lincs.csv')
    print(f"\nLoading DR-Bench data from: {dr_path}")
    dr_df = pd.read_csv(dr_path)
    print(f"  DR-Bench triples: {len(dr_df):,}")
    print(f"  Unique drugs (SMILES): {dr_df['smiles'].nunique():,}")
    
    # Get unique drugs from both benchmarks
    dtr_drugs = dtr_df[['drugId', 'smiles']].drop_duplicates(subset=['smiles'])
    dr_drugs = dr_df[['drugId', 'smiles']].drop_duplicates(subset=['smiles'])
    
    # Union of all drugs (prefer drugId from DTR if available)
    all_drugs = pd.concat([dtr_drugs, dr_drugs]).drop_duplicates(subset=['smiles'], keep='first')
    print(f"\nCombined unique drugs: {len(all_drugs):,}")
    
    # Create SPRINT-compatible CSV
    # SPRINT requires column named exactly "SMILES"
    sprint_drugs = pd.DataFrame({
        'drug_id': all_drugs['drugId'].values,
        'SMILES': all_drugs['smiles'].values  # SPRINT requires this exact column name
    })
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save drugs CSV
    drugs_path = os.path.join(OUTPUT_DIR, 'drugs.csv')
    sprint_drugs.to_csv(drugs_path, index=False)
    print(f"\nSaved: {drugs_path}")
    
    # Save drug_id order for later alignment with embeddings
    drug_ids_path = os.path.join(OUTPUT_DIR, 'drug_ids.txt')
    with open(drug_ids_path, 'w') as f:
        for drug_id in sprint_drugs['drug_id']:
            f.write(f"{drug_id}\n")
    print(f"Saved: {drug_ids_path}")
    
    # Also save SMILES order (more reliable for matching)
    smiles_order_path = os.path.join(OUTPUT_DIR, 'drug_smiles_order.txt')
    with open(smiles_order_path, 'w') as f:
        for smiles in sprint_drugs['SMILES']:
            f.write(f"{smiles}\n")
    print(f"Saved: {smiles_order_path}")
    
    print(f"\n✓ Drug preparation complete: {len(sprint_drugs)} drugs")
    
    return sprint_drugs


# Prepare targets.csv for SPRINT embedding
def fetch_uniprot_sequence(uniprot_id: str, max_retries: int = 3) -> str | None:
    """
    Fetch protein sequence from UniProt REST API.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Parse FASTA - skip header line, join sequence lines
                lines = response.text.strip().split('\n')
                sequence = ''.join(lines[1:])
                return sequence
            elif response.status_code == 404:
                return None
            else:
                time.sleep(1)  # Rate limit backoff
        except requests.exceptions.RequestException:
            time.sleep(1)
    
    return None


def download_alphafold_structure(uniprot_id: str, output_dir: str, max_retries: int = 3) -> str | None:
    """
    Download AlphaFold structure from AlphaFold DB.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded (any version)
    existing = list(out_dir.glob(f"AF-{uniprot_id}-*.cif"))
    if existing:
        return str(existing[0])
    
    for attempt in range(max_retries):
        try:
            # Query AlphaFold DB API for predictions
            preds = list(alphafold_db.get_predictions(uniprot_id))
            if not preds:
                return None
            
            # Prefer fragment F1 (canonical single-chain) when available
            def score_pred(p):
                entry_id = str(p.get("entryId", ""))
                # F1 is the canonical fragment
                return (0 if "-F1" in entry_id else 1, entry_id)
            
            pred = sorted(preds, key=score_pred)[0]
            
            # Download the CIF file using Biopython's helper
            cif_path = alphafold_db.download_cif_for(pred, directory=str(out_dir))
            return cif_path
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                # Log the error for debugging
                print(f"  Warning: Failed to download {uniprot_id}: {e}")
    
    return None


def prepare_targets():
    """
    Prepare targets.csv for SPRINT embedding.
    
    Maps Ensembl Gene IDs -> UniProt accessions -> Protein sequences
    """
    print("\n" + "=" * 80)
    print("PART 2: PREPARING SPRINT TARGET INPUT")
    if USE_STRUCTURE:
        print("(Mode: Structure-aware sequences via AlphaFold + FoldSeek)")
    else:
        print("(Mode: Plain amino acid sequences)")
    print("=" * 80)
    
    # Load DTR-Bench to get target IDs
    dtr_path = os.path.join(DATA_DIR, 'opentargets/drug_target_pairs_csv/drug_target_pairs_lincs.csv')
    dtr_df = pd.read_csv(dtr_path)
    
    # Get unique target IDs (Ensembl Gene IDs)
    target_ids = dtr_df['targetId'].unique().tolist()
    print(f"\nUnique targets (Ensembl Gene IDs): {len(target_ids):,}")
    print(f"Example IDs: {target_ids[:3]}")
    
    # Step 1: Map Ensembl Gene ID -> UniProt accession using mygene.info
    print("\n" + "-" * 40)
    print("Step 1: Mapping Ensembl -> UniProt via mygene.info")
    print("-" * 40)
    
    mg = mygene.MyGeneInfo()
    
    print(f"Querying mygene.info for {len(target_ids)} Ensembl Gene IDs...")
    gene_info = mg.querymany(
        target_ids,
        scopes='ensembl.gene',
        fields='uniprot.Swiss-Prot,symbol',
        species='human',
        as_dataframe=False,
        returnall=True
    )
    
    # Process mygene results
    ensembl_to_uniprot = {}
    ensembl_to_symbol = {}
    unmapped_ensembl = []
    
    for result in gene_info['out']:
        ensembl_id = result.get('query')
        
        if 'notfound' in result:
            unmapped_ensembl.append(ensembl_id)
            continue
        
        # Get gene symbol
        symbol = result.get('symbol')
        if symbol:
            ensembl_to_symbol[ensembl_id] = symbol
        
        # Get UniProt Swiss-Prot accession (reviewed/canonical)
        uniprot_data = result.get('uniprot', {})
        swiss_prot = uniprot_data.get('Swiss-Prot')
        
        if swiss_prot:
            # Can be a list or single value - take first (canonical)
            if isinstance(swiss_prot, list):
                ensembl_to_uniprot[ensembl_id] = swiss_prot[0]
            else:
                ensembl_to_uniprot[ensembl_id] = swiss_prot
        else:
            unmapped_ensembl.append(ensembl_id)
    
    print(f"  Mapped to UniProt: {len(ensembl_to_uniprot):,}")
    print(f"  Unmapped: {len(unmapped_ensembl):,}")
    
    # Save UniProt mapping (useful for both modes)
    mapping_data = []
    for ensembl_id in target_ids:
        uniprot_id = ensembl_to_uniprot.get(ensembl_id)
        symbol = ensembl_to_symbol.get(ensembl_id, '')
        mapping_data.append({
            'ensembl_id': ensembl_id,
            'uniprot_id': uniprot_id,
            'gene_symbol': symbol
        })
    
    mapping_df = pd.DataFrame(mapping_data)
    mapping_path = os.path.join(OUTPUT_DIR, 'uniprot_mapping.csv')
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved: {mapping_path}")
    
    if USE_STRUCTURE:
        # Download AlphaFold structures
        print("\n" + "-" * 40)
        print("Step 2: Downloading AlphaFold structures")
        print("-" * 40)
        
        os.makedirs(STRUCTURES_DIR, exist_ok=True)
        
        unique_uniprot_ids = list(set(ensembl_to_uniprot.values()))
        print(f"Downloading structures for {len(unique_uniprot_ids)} unique UniProt IDs...")
        
        uniprot_to_structure = {}
        failed_structures = []
        
        for uniprot_id in tqdm(unique_uniprot_ids, desc="Downloading"):
            structure_path = download_alphafold_structure(uniprot_id, STRUCTURES_DIR)
            if structure_path:
                uniprot_to_structure[uniprot_id] = structure_path
            else:
                failed_structures.append(uniprot_id)
            
            # Rate limiting
            time.sleep(0.1)
        
        print(f"  Downloaded: {len(uniprot_to_structure):,}")
        print(f"  Failed: {len(failed_structures):,}")
        
        # Create structure file list for FoldSeek processing
        print("\n" + "-" * 40)
        print("Step 3: Creating structure mapping for FoldSeek")
        print("-" * 40)
        
        structure_data = []
        missing_targets = []
        
        for ensembl_id in target_ids:
            uniprot_id = ensembl_to_uniprot.get(ensembl_id)
            symbol = ensembl_to_symbol.get(ensembl_id, '')
            
            if uniprot_id and uniprot_id in uniprot_to_structure:
                structure_data.append({
                    'target_id': ensembl_id,
                    'uniprot_id': uniprot_id,
                    'gene_symbol': symbol,
                    'structure_path': uniprot_to_structure[uniprot_id]
                })
            else:
                missing_targets.append({
                    'ensembl_id': ensembl_id,
                    'uniprot_id': uniprot_id,
                    'symbol': symbol,
                    'reason': 'no_uniprot' if not uniprot_id else 'no_structure'
                })
        
        structure_df = pd.DataFrame(structure_data)
        
        print(f"  Targets with structures: {len(structure_df):,}")
        print(f"  Missing targets: {len(missing_targets):,}")
        if len(target_ids) > 0:
            print(f"  Coverage: {len(structure_df)/len(target_ids)*100:.1f}%")
        
        # Guard: fail early if no structures were obtained
        if structure_df.empty or 'target_id' not in structure_df.columns:
            print("\n" + "=" * 60)
            print("ERROR: No AlphaFold structures were downloaded!")
            print("=" * 60)
            print("Possible causes:")
            print("  1. Network/firewall blocking alphafold.ebi.ac.uk")
            print("  2. UniProt mapping failed (check uniprot_mapping.csv)")
            print("  3. AlphaFold DB doesn't have structures for these proteins")
            print("\nTo debug, try:")
            print("  curl -s 'https://alphafold.ebi.ac.uk/api/prediction/P35367' | head")
            print("\nAlternative: use sequence-only mode (no structure tokens):")
            print("  python 01_prepare_inputs.py --no-structure")
            sys.exit(1)
        
        # Save structure mapping
        structure_map_path = os.path.join(OUTPUT_DIR, 'structure_mapping.csv')
        structure_df.to_csv(structure_map_path, index=False)
        print(f"\nSaved: {structure_map_path}")
        
        # Save target_id order for alignment
        target_ids_path = os.path.join(OUTPUT_DIR, 'target_ids.txt')
        with open(target_ids_path, 'w') as f:
            for target_id in structure_df['target_id']:
                f.write(f"{target_id}\n")
        print(f"Saved: {target_ids_path}")
        
        # Save report
        report = {
            'mode': 'structure-aware',
            'total_ensembl_ids': len(target_ids),
            'mapped_to_uniprot': len(ensembl_to_uniprot),
            'structures_downloaded': len(uniprot_to_structure),
            'final_targets': len(structure_df),
            'coverage_pct': round(len(structure_df)/len(target_ids)*100, 2),
            'unmapped_ensembl': unmapped_ensembl,
            'failed_structure_download': failed_structures,
            'missing_targets': missing_targets
        }
        
        report_path = os.path.join(OUTPUT_DIR, 'mapping_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved: {report_path}")
        
        print(f"\n✓ Structure preparation complete: {len(structure_df)} targets")
        print(f"\n" + "=" * 80)
        print("NEXT STEP: Generate structure-aware sequences with FoldSeek")
        print("=" * 80)
        print(f"""
Run the following in the SPRINT repo (panspecies-dti):

    cd /path/to/panspecies-dti
    
    # For each structure file, run:
    python utils/structure_to_saprot.py \\
        -I {STRUCTURES_DIR}/AF-<UNIPROT_ID>-F1-model_v4.cif \\
        --chain A \\
        -O {OUTPUT_DIR}/targets_foldseek.csv
    
    # Or use the helper script we provide:
    bash {os.path.dirname(os.path.abspath(__file__))}/02a_run_foldseek.sh

After generating targets_foldseek.csv, rename it to targets.csv:
    mv {OUTPUT_DIR}/targets_foldseek.csv {OUTPUT_DIR}/targets.csv
""")
        
        return structure_df
        
    else:
        # Step 2: Fetch protein sequences from UniProt
        print("\n" + "-" * 40)
        print("Step 2: Fetching protein sequences from UniProt")
        print("-" * 40)
        
        unique_uniprot_ids = list(set(ensembl_to_uniprot.values()))
        print(f"Fetching sequences for {len(unique_uniprot_ids)} unique UniProt IDs...")
        
        uniprot_to_sequence = {}
        failed_uniprot = []
        
        for uniprot_id in tqdm(unique_uniprot_ids, desc="Fetching"):
            sequence = fetch_uniprot_sequence(uniprot_id)
            if sequence:
                uniprot_to_sequence[uniprot_id] = sequence
            else:
                failed_uniprot.append(uniprot_id)
            
            time.sleep(0.1)
        
        print(f"  Fetched: {len(uniprot_to_sequence):,}")
        print(f"  Failed: {len(failed_uniprot):,}")
        
        # Step 3: Create targets CSV
        print("\n" + "-" * 40)
        print("Step 3: Creating SPRINT targets CSV")
        print("-" * 40)
        
        target_data = []
        missing_targets = []
        
        for ensembl_id in target_ids:
            uniprot_id = ensembl_to_uniprot.get(ensembl_id)
            symbol = ensembl_to_symbol.get(ensembl_id, '')
            
            if uniprot_id and uniprot_id in uniprot_to_sequence:
                sequence = uniprot_to_sequence[uniprot_id]
                target_data.append({
                    'target_id': ensembl_id,
                    'uniprot_id': uniprot_id,
                    'gene_symbol': symbol,
                    'Target Sequence': sequence
                })
            else:
                missing_targets.append({
                    'ensembl_id': ensembl_id,
                    'uniprot_id': uniprot_id,
                    'symbol': symbol,
                    'reason': 'no_uniprot' if not uniprot_id else 'no_sequence'
                })
        
        sprint_targets = pd.DataFrame(target_data)
        
        print(f"  Targets with sequences: {len(sprint_targets):,}")
        print(f"  Missing targets: {len(missing_targets):,}")
        print(f"  Coverage: {len(sprint_targets)/len(target_ids)*100:.1f}%")
        
        # Save targets CSV
        targets_path = os.path.join(OUTPUT_DIR, 'targets.csv')
        sprint_targets.to_csv(targets_path, index=False)
        print(f"\nSaved: {targets_path}")
        
        # Save target_id order for alignment
        target_ids_path = os.path.join(OUTPUT_DIR, 'target_ids.txt')
        with open(target_ids_path, 'w') as f:
            for target_id in sprint_targets['target_id']:
                f.write(f"{target_id}\n")
        print(f"Saved: {target_ids_path}")
        
        # Save report
        report = {
            'mode': 'plain-sequence',
            'total_ensembl_ids': len(target_ids),
            'mapped_to_uniprot': len(ensembl_to_uniprot),
            'sequences_fetched': len(uniprot_to_sequence),
            'final_targets': len(sprint_targets),
            'coverage_pct': round(len(sprint_targets)/len(target_ids)*100, 2),
            'unmapped_ensembl': unmapped_ensembl,
            'failed_uniprot_fetch': failed_uniprot,
            'missing_targets': missing_targets
        }
        
        report_path = os.path.join(OUTPUT_DIR, 'mapping_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved: {report_path}")
        
        print(f"\n✓ Target preparation complete: {len(sprint_targets)} targets")
        print("\nNote: Using plain sequences. SPRINT will use mask tokens for structure.")
        print("For better performance, run without --no-structure flag.")
        
        return sprint_targets


if __name__ == '__main__':
    print("=" * 80)
    print("SPRINT INPUT PREPARATION")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Mode: {'Structure-aware (AlphaFold + FoldSeek)' if USE_STRUCTURE else 'Plain sequences'}")
    
    # Prepare drugs
    drugs_df = prepare_drugs()
    
    # Prepare targets
    targets_df = prepare_targets()
    
    # Summary
    print("\n" + "=" * 80)
    print("PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs in: {OUTPUT_DIR}/")
    print(f"  drugs.csv              - {len(drugs_df)} drugs with SMILES")
    
    if USE_STRUCTURE:
        print(f"  structure_mapping.csv  - {len(targets_df)} targets with structure paths")
        print(f"  alphafold_structures/  - Downloaded CIF files")
        print(f"  uniprot_mapping.csv    - Ensembl -> UniProt mapping")
        print(f"  target_ids.txt         - Target ID order for alignment")
        print(f"  mapping_report.json    - Coverage report")