import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings
import concurrent.futures
from functools import partial
import pickle
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BRDtoChEMBLMapper:
    def __init__(self, cache_dir: str = "./cache", use_persistent_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_persistent_cache = use_persistent_cache
        self.cache_file = self.cache_dir / "inchi_chembl_cache.pkl"
        self.inchi_to_chembl_cache = self._load_cache()
        self.chembl_client = None
        self.setup_chembl_client()
        
    def setup_chembl_client(self):
        try:
            from chembl_webresource_client.new_client import new_client
            self.chembl_client = new_client
            logger.info("ChEMBL client initialized successfully")
        except ImportError:
            logger.error("ChEMBL client not available. Install with: pip install chembl_webresource_client")
            raise
    
    def _load_cache(self) -> Dict:
        if self.use_persistent_cache and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    logger.info(f"Loaded {len(cache)} cached InChI-ChEMBL mappings")
                    return cache
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        if self.use_persistent_cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.inchi_to_chembl_cache, f)
                logger.info(f"Saved {len(self.inchi_to_chembl_cache)} mappings to cache")
            except Exception as e:
                logger.warning(f"Could not save cache: {e}")
    
    def load_lincs_metadata(self, metadata_file: Optional[str] = None) -> pd.DataFrame:
        if metadata_file and Path(metadata_file).exists():
            logger.info(f"Loading metadata from {metadata_file}")
            df = pd.read_csv(metadata_file, sep='\t')
            logger.info(f"Loaded metadata with columns: {df.columns.tolist()[:10]}...")
            return df
        
        cache_file = self.cache_dir / "GSE92742_Broad_LINCS_pert_info.txt"
        
        if cache_file.exists():
            logger.info(f"Loading cached metadata from {cache_file}")
            df = pd.read_csv(cache_file, sep='\t')
            logger.info(f"Loaded metadata with columns: {df.columns.tolist()[:10]}...")
            return df
        geo_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file=GSE92742%5FBroad%5FLINCS%5Fpert%5Finfo%2Etxt%2Egz"
        
        try:
            logger.info("Downloading LINCS metadata from GEO...")
            import gzip
            import urllib.request
            
            with urllib.request.urlopen(geo_url) as response:
                with gzip.open(response, 'rt') as gz:
                    df = pd.read_csv(gz, sep='\t')
                    df.to_csv(cache_file, sep='\t', index=False)
                    logger.info(f"Metadata saved to {cache_file}")
                    logger.info(f"Loaded metadata with columns: {df.columns.tolist()[:10]}...")
                    return df
        except Exception as e:
            logger.error(f"Failed to download metadata: {e}")
            raise
    
    def _query_chembl_single(self, inchi_key: str) -> Optional[str]:
        """
        query ChemBL for a inchi key.
        """
        if not inchi_key or pd.isna(inchi_key):
            return None
        if inchi_key in self.inchi_to_chembl_cache:
            return self.inchi_to_chembl_cache[inchi_key]
        try:
            molecule_api = self.chembl_client.molecule
            mols = molecule_api.filter(
                molecule_structures__standard_inchi_key=inchi_key
            ).only(['molecule_chembl_id'])
            
            if mols:
                chembl_id = mols[0]['molecule_chembl_id']
                self.inchi_to_chembl_cache[inchi_key] = chembl_id
                return chembl_id
        except Exception as e:
            logger.debug(f"Failed to map InChI key {inchi_key}: {e}")
        self.inchi_to_chembl_cache[inchi_key] = None
        return None
    
    def _parallel_query_chembl(self, inchi_keys: List[str], max_workers: int = 10) -> Dict[str, str]:
        """
        query ChemBL for multiple inchi keys.
        """
        uncached_keys = [k for k in inchi_keys 
                         if k and pd.notna(k) and k not in self.inchi_to_chembl_cache]
        
        if not uncached_keys:
            logger.info("All InChI keys found in cache!")
            return {k: self.inchi_to_chembl_cache.get(k) for k in inchi_keys if k}
        
        logger.info(f"Querying ChEMBL for {len(uncached_keys)} uncached InChI keys using {max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {executor.submit(self._query_chembl_single, key): key 
                           for key in uncached_keys}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_key)):
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(uncached_keys)} InChI keys")
                try:
                    future.result()
                except Exception as e:
                    logger.debug(f"Error processing key: {e}")
        self._save_cache()
        
        return {k: self.inchi_to_chembl_cache.get(k) for k in inchi_keys if k}
    
    def map_brd_to_chembl(
        self,
        brd_ids: Union[List[str], pd.Series],
        metadata_df: Optional[pd.DataFrame] = None,
        max_workers: int = 10
    ) -> pd.DataFrame:
        """
        Map from BRD IDs to ChEMBL IDs.
        """
        if isinstance(brd_ids, pd.Series):
            brd_ids = brd_ids.tolist()
        unique_brds = list(dict.fromkeys(brd_ids))
        logger.info(f"Processing {len(unique_brds)} unique BRD IDs")
        
        if metadata_df is None:
            metadata_df = self.load_lincs_metadata()
        brd_df = pd.DataFrame({'brd_id': unique_brds})
        brd_df['core_brd'] = brd_df['brd_id'].str[:13]
        
        available_cols = ['pert_id', 'inchi_key']
        if 'pert_name' in metadata_df.columns:
            available_cols.append('pert_name')

        merged = brd_df.merge(
            metadata_df[available_cols].drop_duplicates('pert_id'),
            left_on='core_brd',
            right_on='pert_id',
            how='left'
        )
        
        if 'pert_name' not in merged.columns:
            merged['pert_name'] = ''
        
        unique_inchi_keys = merged['inchi_key'].dropna().unique().tolist()
        logger.info(f"Found {len(unique_inchi_keys)} unique InChI keys")
        inchi_to_chembl = self._parallel_query_chembl(unique_inchi_keys, max_workers)
        
        merged['chembl_id'] = merged['inchi_key'].map(inchi_to_chembl)
        merged['confidence'] = merged['chembl_id'].notna().astype(float) * 0.95
        
        result_cols = ['brd_id', 'chembl_id', 'confidence']
        if 'pert_name' in merged.columns:
            result_cols.append('pert_name')
            results = merged[result_cols].copy()
            results['pert_name'] = results['pert_name'].fillna('')
        else:
            results = merged[result_cols].copy()
            results['pert_name'] = ''
        
        return results
    
    def process_dataframe(self, df: pd.DataFrame, pert_id_column: str = 'pert_id',
                          max_workers: int = 10) -> pd.DataFrame:
        """
        Process entire dataframe efficiently.
        """
        # get unique BRD IDs
        unique_brds = df[pert_id_column].unique()
        
        # load metadata
        metadata_df = self.load_lincs_metadata()
        
        # map all at once
        mappings = self.map_brd_to_chembl(unique_brds, metadata_df, max_workers)
        
        # create mapping dictionaries
        brd_to_chembl = dict(zip(mappings['brd_id'], mappings['chembl_id']))
        brd_to_confidence = dict(zip(mappings['brd_id'], mappings['confidence']))
        
        # vectorized assignment
        df['chembl_id'] = df[pert_id_column].map(brd_to_chembl)
        df['mapping_confidence'] = df[pert_id_column].map(brd_to_confidence)
        
        return df
    
    def generate_summary_report(self, mappings_df: pd.DataFrame) -> Dict:
        """Generate summary statistics."""
        total = len(mappings_df)
        mapped = mappings_df['chembl_id'].notna().sum()
        
        return {
            'total_brd_ids': total,
            'successfully_mapped': mapped,
            'mapping_rate': f"{(mapped/total)*100:.1f}%" if total > 0 else "0%",
            'cache_size': len(self.inchi_to_chembl_cache)
        }


def main(): 
    data_dir = os.environ['CONTEXTPERT_DATA_DIR']
    input_path = os.path.join(data_dir, "full_lincs.csv")
    output_path = os.path.join(data_dir, "full_lincs_with_chembl.csv")
    
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    mapper = BRDtoChEMBLMapper(
        cache_dir='./brd_chembl_cache',
        use_persistent_cache=True
    )
    logger.info("Starting fast BRD to ChEMBL mapping...")
    df = mapper.process_dataframe(
        df, 
        pert_id_column='pert_id',
        max_workers=20
    )
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")
    
    # Generate summary report
    unique_brds = df['pert_id'].nunique()
    mapped = df['chembl_id'].notna().sum()
    logger.info(f"\n=== Final Summary ===")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Unique BRDs: {unique_brds}")
    logger.info(f"Successfully mapped: {mapped}/{len(df)} ({mapped/len(df)*100:.1f}%)")
    logger.info(f"Cache size: {len(mapper.inchi_to_chembl_cache)} InChI-ChEMBL pairs")
    
    return df


if __name__ == "__main__":
    main()