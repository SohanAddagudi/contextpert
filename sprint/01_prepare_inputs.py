import os
import json
import pandas as pd
import mygene
import requests
from tqdm import tqdm
import time
from pathlib import Path

DATA_DIR = os.environ.get('CONTEXTPERT_DATA_DIR')
if not DATA_DIR:
    raise ValueError("Please set CONTEXTPERT_DATA_DIR environment variable")

OUTPUT_DIR = os.path.join(DATA_DIR, 'sprint')

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


def prepare_targets():
    """
    Prepare targets.csv for SPRINT embedding.
    
    Maps Ensembl Gene IDs -> UniProt accessions -> Protein sequences
    """
    print("\n" + "=" * 80)
    print("PART 2: PREPARING SPRINT TARGET INPUT")
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
    
    # Step 2: Fetch protein sequences from UniProt
    print("\n" + "-" * 40)
    print("Step 2: Fetching protein sequences from UniProt")
    print("-" * 40)
    
    # Get unique UniProt IDs
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
        
        # Rate limiting (UniProt recommends ~10 req/sec for unauthenticated)
        time.sleep(0.1)
    
    print(f"  Fetched: {len(uniprot_to_sequence):,}")
    print(f"  Failed: {len(failed_uniprot):,}")
    
    # Step 3: Create final targets CSV
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
                'Target Sequence': sequence  # SPRINT requires this exact column name
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
    
    # Save mapping report
    report = {
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
    
    return sprint_targets


if __name__ == '__main__':
    print("=" * 80)
    print("SPRINT INPUT PREPARATION")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Prepare drugs
    drugs_df = prepare_drugs()
    
    # Prepare targets
    targets_df = prepare_targets()
    
    # Summary
    print("\n" + "=" * 80)
    print("PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs in: {OUTPUT_DIR}/")
    print(f"  drugs.csv          - {len(drugs_df)} drugs with SMILES")
    print(f"  targets.csv        - {len(targets_df)} targets with sequences")
    print(f"  drug_ids.txt       - Drug ID order for alignment")
    print(f"  target_ids.txt     - Target ID order for alignment")
    print(f"  mapping_report.json - Coverage report")