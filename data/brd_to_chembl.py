import pandas as pd
import numpy as np
import requests
import json
import time
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import sqlite3
from rdkit import Chem
from rdkit.Chem import inchi
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BRDtoChEMBLMapper:
    
    def __init__(self, cache_dir: str = "./cache"):

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize cache databases
        self.mapping_cache = {}
        self.chembl_client = None
        self.setup_chembl_client()
        
    def setup_chembl_client(self):
        """Set up ChEMBL web resource client."""
        try:
            from chembl_webresource_client.new_client import new_client
            self.chembl_client = new_client
            logger.info("ChEMBL client initialized successfully")
        except ImportError:
            logger.warning("ChEMBL client not available. Install with: pip install chembl_webresource_client")
    
    # using the table from GSE92742
    
    def load_lincs_metadata(self, metadata_file: Optional[str] = None) -> pd.DataFrame:
        if metadata_file and Path(metadata_file).exists():
            logger.info(f"Loading metadata from {metadata_file}")
            return pd.read_csv(metadata_file, sep='\t')
        
        # Try to download from GEO
        geo_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file=GSE92742%5FBroad%5FLINCS%5Fpert%5Finfo%2Etxt%2Egz"
        cache_file = self.cache_dir / "GSE92742_Broad_LINCS_pert_info.txt"
        
        if cache_file.exists():
            logger.info(f"Loading cached metadata from {cache_file}")
            return pd.read_csv(cache_file, sep='\t')
        
        try:
            logger.info("Downloading LINCS metadata from GEO...")
            import gzip
            import urllib.request
            
            with urllib.request.urlopen(geo_url) as response:
                with gzip.open(response, 'rt') as gz:
                    df = pd.read_csv(gz, sep='\t')
                    df.to_csv(cache_file, sep='\t', index=False)
                    logger.info(f"Metadata saved to {cache_file}")
                    return df
        except Exception as e:
            logger.error(f"Failed to download metadata: {e}")
            return pd.DataFrame()
    
    # not using this method
    def map_via_precomputed_table(self, brd_ids: List[str]) -> pd.DataFrame:
        # Load custom mapping table (example structure)
        mapping_file = self.cache_dir / "brd_chembl_mapping.csv"
        
        if mapping_file.exists():
            mapping_df = pd.read_csv(mapping_file)
            results = []
            
            for brd_id in brd_ids:
                # Extract core BRD (first 13 characters)
                core_brd = brd_id[:13] if len(brd_id) >= 13 else brd_id
                
                match = mapping_df[mapping_df['brd_id'].str.startswith(core_brd)]
                if not match.empty:
                    for _, row in match.iterrows():
                        results.append({
                            'brd_id': brd_id,
                            'chembl_id': row.get('chembl_id', None),
                            'confidence': row.get('confidence', 0.8),
                            'method': 'precomputed_table'
                        })
                else:
                    results.append({
                        'brd_id': brd_id,
                        'chembl_id': None,
                        'confidence': 0,
                        'method': 'precomputed_table'
                    })
            
            return pd.DataFrame(results)
        else:
            logger.warning("No precomputed mapping table found")
            return pd.DataFrame()
    
    # structure-based mapping -> using InChI/SMILES to find ChEMBL IDs
    
    def get_compound_structure(self, brd_id: str, metadata_df: pd.DataFrame) -> Dict:
        core_brd = brd_id[:13] if len(brd_id) >= 13 else brd_id
        
        compound = metadata_df[metadata_df['pert_id'].str.startswith(core_brd)]
        
        if compound.empty:
            return {}
        
        row = compound.iloc[0]
        return {
            'smiles': row.get('canonical_smiles', None),
            'inchi': row.get('inchi', None),
            'inchi_key': row.get('inchi_key', None)
        }
    
    def map_via_structure(self, brd_ids: List[str], metadata_df: pd.DataFrame) -> pd.DataFrame:
        if self.chembl_client is None:
            logger.error("ChEMBL client not available")
            return pd.DataFrame()
        
        results = []
        molecule_api = self.chembl_client.molecule
        
        for brd_id in brd_ids:
            structure_info = self.get_compound_structure(brd_id, metadata_df)
            
            if not structure_info:
                results.append({
                    'brd_id': brd_id,
                    'chembl_id': None,
                    'confidence': 0,
                    'method': 'structure'
                })
                continue
            
            chembl_id = None
            confidence = 0
            
            # Try InChI Key first (most specific)
            if structure_info.get('inchi_key'):
                try:
                    mols = molecule_api.filter(
                        molecule_structures__standard_inchi_key=structure_info['inchi_key']
                    )
                    if mols:
                        chembl_id = mols[0]['molecule_chembl_id']
                        confidence = 0.95
                except Exception as e:
                    logger.debug(f"InChI key search failed for {brd_id}: {e}")
            
            # Try SMILES if InChI didn't work
            if not chembl_id and structure_info.get('smiles'):
                try:
                    # Use similarity search with high threshold
                    mols = molecule_api.filter(
                        molecule_structures__canonical_smiles__flexmatch=structure_info['smiles']
                    )
                    if mols:
                        chembl_id = mols[0]['molecule_chembl_id']
                        confidence = 0.9
                except Exception as e:
                    logger.debug(f"SMILES search failed for {brd_id}: {e}")
            
            results.append({
                'brd_id': brd_id,
                'chembl_id': chembl_id,
                'confidence': confidence,
                'method': 'structure'
            })
        
        return pd.DataFrame(results)
    
    # pubchem intermediate mapping -> using PubChem CIDs to find ChEMBL IDs
    
    def get_pubchem_cid(self, brd_id: str, metadata_df: pd.DataFrame) -> Optional[int]:
        core_brd = brd_id[:13] if len(brd_id) >= 13 else brd_id
        compound = metadata_df[metadata_df['pert_id'].str.startswith(core_brd)]
        
        if not compound.empty:
            cid = compound.iloc[0].get('pubchem_cid', None)
            if pd.notna(cid):
                return int(cid)
        return None
    
    def map_pubchem_to_chembl(self, pubchem_cids: List[int]) -> Dict[int, str]:
        if self.chembl_client is None:
            return {}
        
        mapping = {}
        molecule_api = self.chembl_client.molecule
        
        for cid in pubchem_cids:
            try:
                # Search ChEMBL for PubChem cross-reference
                mols = molecule_api.filter(
                    cross_references__xref_id=str(cid),
                    cross_references__xref_src="PubChem"
                )
                if mols:
                    mapping[cid] = mols[0]['molecule_chembl_id']
            except Exception as e:
                logger.debug(f"Failed to map PubChem CID {cid}: {e}")
        
        return mapping
    
    def map_via_pubchem(self, brd_ids: List[str], metadata_df: pd.DataFrame) -> pd.DataFrame:
        results = []
        
        # Get PubChem CIDs for all BRD IDs
        brd_to_cid = {}
        cids = []
        
        for brd_id in brd_ids:
            cid = self.get_pubchem_cid(brd_id, metadata_df)
            if cid:
                brd_to_cid[brd_id] = cid
                cids.append(cid)
        
        # Map PubChem CIDs to ChEMBL
        cid_to_chembl = self.map_pubchem_to_chembl(cids)
        
        # Compile results
        for brd_id in brd_ids:
            cid = brd_to_cid.get(brd_id)
            chembl_id = cid_to_chembl.get(cid) if cid else None
            
            results.append({
                'brd_id': brd_id,
                'pubchem_cid': cid,
                'chembl_id': chembl_id,
                'confidence': 0.85 if chembl_id else 0,
                'method': 'pubchem_intermediate'
            })
        
        return pd.DataFrame(results)
    
    # clue api
    
    def query_clue_api(self, brd_ids: List[str], api_key: Optional[str] = None) -> pd.DataFrame:
        base_url = "https://api.clue.io/api/perts"
        results = []
        
        for brd_id in brd_ids:
            core_brd = brd_id[:13] if len(brd_id) >= 13 else brd_id
            
            params = {
                'filter': json.dumps({'where': {'pert_id': core_brd}}),
                'user_key': api_key
            } if api_key else {
                'filter': json.dumps({'where': {'pert_id': core_brd}})
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        compound = data[0]
                        results.append({
                            'brd_id': brd_id,
                            'pert_name': compound.get('pert_name'),
                            'pubchem_cid': compound.get('pubchem_cid'),
                            'inchi_key': compound.get('inchi_key'),
                            'moa': compound.get('moa', [])
                        })
                        continue
            except Exception as e:
                logger.debug(f"CLUE API query failed for {brd_id}: {e}")
            
            results.append({
                'brd_id': brd_id,
                'pert_name': None,
                'pubchem_cid': None,
                'inchi_key': None,
                'moa': []
            })
        
        return pd.DataFrame(results)
    
    # main mapping function
    
    def map_brd_to_chembl(
        self,
        brd_ids: Union[List[str], pd.Series],
        methods: List[str] = ['structure', 'pubchem', 'precomputed'],
        metadata_file: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Main function to map BRD IDs to ChEMBL IDs using multiple methods.
        
        Parameters:
        -----------
        brd_ids : List[str] or pd.Series
            BRD identifiers to map
        methods : List[str]
            Methods to use: 'structure', 'pubchem', 'precomputed', 'clue'
        metadata_file : str, optional
            Path to LINCS metadata file
        api_key : str, optional
            CLUE API key
        
        Returns:
        --------
        pd.DataFrame
            Comprehensive mapping results with columns:
            - brd_id: Original BRD identifier
            - chembl_id: Mapped ChEMBL ID
            - confidence: Confidence score (0-1)
            - method: Method used for mapping
            - pubchem_cid: PubChem CID if available
            - pert_name: Compound name
        """
        if isinstance(brd_ids, pd.Series):
            brd_ids = brd_ids.tolist()
        
        # Remove duplicates while preserving order
        brd_ids = list(dict.fromkeys(brd_ids))
        
        logger.info(f"Mapping {len(brd_ids)} unique BRD IDs to ChEMBL")
        
        # Load metadata
        metadata_df = self.load_lincs_metadata(metadata_file)
        
        # Initialize results
        all_results = []
        
        # Try each method
        if 'precomputed' in methods:
            logger.info("Trying precomputed table method...")
            results = self.map_via_precomputed_table(brd_ids)
            if not results.empty:
                all_results.append(results)
        
        if 'structure' in methods and not metadata_df.empty:
            logger.info("Trying structure-based mapping...")
            results = self.map_via_structure(brd_ids, metadata_df)
            if not results.empty:
                all_results.append(results)
        
        if 'pubchem' in methods and not metadata_df.empty:
            logger.info("Trying PubChem intermediate mapping...")
            results = self.map_via_pubchem(brd_ids, metadata_df)
            if not results.empty:
                all_results.append(results)
        
        if 'clue' in methods:
            logger.info("Querying CLUE API...")
            clue_results = self.query_clue_api(brd_ids, api_key)
            # Note: CLUE results need additional processing to get ChEMBL IDs
        
        # Combine results, keeping best confidence for each BRD
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            
            # Group by BRD ID and keep highest confidence mapping
            idx = combined.groupby('brd_id')['confidence'].idxmax()
            final_results = combined.loc[idx].reset_index(drop=True)
            
            # Add compound names from metadata if available
            if not metadata_df.empty:
                for i, row in final_results.iterrows():
                    if pd.isna(row.get('pert_name')):
                        core_brd = row['brd_id'][:13] if len(row['brd_id']) >= 13 else row['brd_id']
                        compound = metadata_df[metadata_df['pert_id'].str.startswith(core_brd)]
                        if not compound.empty:
                            final_results.at[i, 'pert_name'] = compound.iloc[0].get('pert_name', '')
            
            return final_results
        else:
            logger.warning("No mapping results obtained")
            return pd.DataFrame(columns=['brd_id', 'chembl_id', 'confidence', 'method'])
    
    def save_mappings(self, mappings_df: pd.DataFrame, output_file: str):
        output_path = Path(output_file)
        
        if output_path.suffix == '.xlsx':
            mappings_df.to_excel(output_file, index=False)
        else:
            mappings_df.to_csv(output_file, index=False)
        
        logger.info(f"Mappings saved to {output_file}")
    
    def generate_summary_report(self, mappings_df: pd.DataFrame) -> Dict:
        total = len(mappings_df)
        mapped = mappings_df['chembl_id'].notna().sum()
        
        summary = {
            'total_brd_ids': total,
            'successfully_mapped': mapped,
            'mapping_rate': f"{(mapped/total)*100:.1f}%" if total > 0 else "0%",
            'methods_used': mappings_df['method'].value_counts().to_dict(),
            'average_confidence': mappings_df[mappings_df['chembl_id'].notna()]['confidence'].mean(),
            'high_confidence_mappings': (mappings_df['confidence'] >= 0.9).sum()
        }
        
        return summary


# test usage

def main():
    sample_brd_ids = [
        "BRD-A85280935",
        "BRD-K23355843",
        "BRD-A37704979",
        "BRD-A44008656",
        "BRD-K22134346",
        "BRD-K02404261"
    ]
    
    # Initialize mapper
    mapper = BRDtoChEMBLMapper(cache_dir="./brd_chembl_cache")
    
    # Perform mapping using multiple methods
    results = mapper.map_brd_to_chembl(
        brd_ids=sample_brd_ids,
        methods=['structure', 'pubchem'],  # You can add 'clue' if you have API key
        metadata_file=None,  # Will attempt to download from GEO
        api_key=None  # Add your CLUE API key if available
    )
    
    # Display results
    print("\n=== Mapping Results ===")
    print(results.to_string())
    
    # Generate summary
    summary = mapper.generate_summary_report(results)
    print("\n=== Summary Report ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save results
    mapper.save_mappings(results, "brd_to_chembl_mappings.csv")
    
    return results

# make sure get only unique BRD IDs
def process_lincs_dataset(df: pd.DataFrame, pert_id_column: str = 'brd_id') -> pd.DataFrame:
    unique_brds = df[pert_id_column].unique()
    mapper = BRDtoChEMBLMapper()
    mappings = mapper.map_brd_to_chembl(unique_brds)
    brd_to_chembl = dict(zip(mappings['brd_id'], mappings['chembl_id']))
    df['chembl_id'] = df[pert_id_column].map(brd_to_chembl)
    brd_to_confidence = dict(zip(mappings['brd_id'], mappings['confidence']))
    df['mapping_confidence'] = df[pert_id_column].map(brd_to_confidence)
    
    return df


def validate_mappings(mappings_df: pd.DataFrame) -> pd.DataFrame:
    try:
        from chembl_webresource_client.new_client import new_client
        molecule_api = new_client.molecule
        
        validation_results = []
        
        for _, row in mappings_df.iterrows():
            if pd.notna(row['chembl_id']):
                try:
                    mol = molecule_api.get(row['chembl_id'])
                    valid = mol is not None
                    mol_name = mol.get('pref_name', '') if mol else ''
                except:
                    valid = False
                    mol_name = ''
            else:
                valid = False
                mol_name = ''
            
            validation_results.append({
                'valid': valid,
                'chembl_name': mol_name
            })
        
        validation_df = pd.DataFrame(validation_results)
        return pd.concat([mappings_df, validation_df], axis=1)
    
    except ImportError:
        logger.warning("Cannot validate without ChEMBL client")
        mappings_df['valid'] = None
        mappings_df['chembl_name'] = None
        return mappings_df


if __name__ == "__main__":
    # input_path = (
    #     '/mmfs1/gscratch/ark/jiaqi/projects/context/contextpert_new/data/merged_output4_head.csv'
    # )
    input_path = (
        '/mmfs1/gscratch/ark/jiaqi/projects/context/contextpert_new/data/merged_output4_head.csv'
    )
    df = pd.read_csv(input_path)

    # inst_info_path = (
    #     '/mmfs1/gscratch/ark/jiaqi/projects/cml/ot_data/GSE92742_Broad_LINCS_inst_info.txt'
    # )

    mapper = BRDtoChEMBLMapper(cache_dir='./brd_chembl_cache')

    print(df['pert_id'].unique()[:10])

    results = mapper.map_brd_to_chembl(
        brd_ids=df['pert_id'],
        methods=['structure','pubchem'],
        metadata_file=None,
        api_key=None
    )

    """
    print("\n=== Mapping Results ===")
    print(results.to_string())
    print("\n=== Summary Report ===")
    summary = mapper.generate_summary_report(results)
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(results['chembl_id'].unique()[:10])  # Display first 10 unique ChEMBL IDs
    """

    df = df.merge(
        results[['brd_id','chembl_id','confidence']],
        left_on='pert_id', right_on='brd_id', how='left'
    )

    out_path = (
        '/mmfs1/gscratch/ark/jiaqi/projects/context/contextpert_new/data/merged_output4_with_chembl.csv'
    )
    df.to_csv(out_path, index=False)
    print(f"Wrote annotated data to {out_path}")
