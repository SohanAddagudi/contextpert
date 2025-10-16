"""
ChEMBL utilities for batch conversion of ChEMBL IDs to SMILES with caching,
and BRD to ChEMBL ID mapping
"""
import os
import pickle
import warnings
import logging
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
from typing import List, Dict, Optional, Union
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.chembl_cache')
DEFAULT_BRD_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.brd_chembl_cache')


def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form using RDKit

    Args:
        smiles: SMILES string to canonicalize

    Returns:
        Canonical SMILES string

    Raises:
        ValueError: If SMILES string is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


class ChEMBLCache:
    """Persistent cache for ChEMBL ID to SMILES mappings"""

    def __init__(self, cache_dir=None):
        """Initialize cache

        Args:
            cache_dir: Directory to store cache file. If None, uses default.
        """
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'chembl_smiles_cache.pkl'
        self._cache = self._load_cache()

    def _load_cache(self):
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    if isinstance(cache, dict):
                        return cache
            except Exception as e:
                warnings.warn(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            warnings.warn(f"Failed to save cache: {e}")

    def get(self, chembl_id):
        """Get SMILES for a ChEMBL ID from cache"""
        return self._cache.get(chembl_id)

    def set(self, chembl_id, smiles):
        """Set SMILES for a ChEMBL ID in cache"""
        self._cache[chembl_id] = smiles

    def __contains__(self, chembl_id):
        """Check if ChEMBL ID is in cache"""
        return chembl_id in self._cache

    def save(self):
        """Explicitly save cache to disk"""
        self._save_cache()

    def size(self):
        """Return number of entries in cache"""
        return len(self._cache)


# Global cache instance
_CACHE = None


def get_cache(cache_dir=None):
    """Get or create the global cache instance"""
    global _CACHE
    if _CACHE is None:
        _CACHE = ChEMBLCache(cache_dir=cache_dir)
    return _CACHE


def chembl_to_smiles_batch(chembl_ids, batch_size=50, show_progress=True, cache_dir=None):
    """Batch convert ChEMBL IDs to canonical SMILES with progress tracking

    Uses a persistent offline cache to minimize API calls. Results are automatically
    saved to disk and reloaded in future runs.

    Args:
        chembl_ids: List or array of ChEMBL identifiers
        batch_size: Number of ChEMBL IDs to fetch per API request (default: 50)
        show_progress: Whether to show tqdm progress bar (default: True)
        cache_dir: Directory for cache file. If None, uses default (.chembl_cache)

    Returns:
        dict: Mapping from ChEMBL ID to SMILES (None if not found)

    Example:
        >>> ids = ['CHEMBL1200699', 'CHEMBL1096882']
        >>> results = chembl_to_smiles_batch(ids)
        >>> print(results['CHEMBL1200699'])
        'C[C@H]1c2cccc(O)c2C(=O)C2=C(O)[C@]3(O)...'
    """
    # Get cache first
    cache = get_cache(cache_dir)

    # Filter to uncached IDs
    chembl_ids = list(chembl_ids)
    uncached_ids = [cid for cid in chembl_ids if cid not in cache]

    if not uncached_ids:
        # All cached, return immediately without loading ChEMBL client
        if show_progress:
            print(f"Cache: {len(chembl_ids)}/{len(chembl_ids)} IDs cached (all from cache, no API calls needed)")
        return {cid: cache.get(cid) for cid in chembl_ids}

    # Only import ChEMBL client if we need to fetch from API
    from chembl_webresource_client.new_client import new_client
    molecule = new_client.molecule

    if show_progress:
        print(f"Cache: {len(chembl_ids) - len(uncached_ids)}/{len(chembl_ids)} IDs cached, "
              f"fetching {len(uncached_ids)} from API")

    # Process in batches
    iterator = range(0, len(uncached_ids), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Fetching SMILES",
                       total=(len(uncached_ids) + batch_size - 1) // batch_size,
                       unit="batch")

    for i in iterator:
        batch = uncached_ids[i:i + batch_size]

        try:
            # Batch query using filter
            results = molecule.filter(molecule_chembl_id__in=batch).only(
                ['molecule_chembl_id', 'molecule_structures']
            )

            # Process results
            found_ids = set()
            for result in results:
                chembl_id = result.get('molecule_chembl_id')
                if chembl_id and 'molecule_structures' in result and result['molecule_structures']:
                    smiles = result['molecule_structures'].get('canonical_smiles')
                    cache.set(chembl_id, smiles)
                    found_ids.add(chembl_id)

            # Mark unfound IDs as None
            for chembl_id in batch:
                if chembl_id not in found_ids:
                    cache.set(chembl_id, None)

        except Exception as e:
            warnings.warn(f"Batch query failed for batch starting at {i}: {e}")
            # Mark all as None for this failed batch
            for chembl_id in batch:
                if chembl_id not in cache:
                    cache.set(chembl_id, None)

    # Save cache to disk
    cache.save()

    return {cid: cache.get(cid) for cid in chembl_ids}


def clear_cache(cache_dir=None):
    """Clear the ChEMBL SMILES cache

    Args:
        cache_dir: Directory containing cache file. If None, uses default.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_file = Path(cache_dir) / 'chembl_smiles_cache.pkl'
    if cache_file.exists():
        cache_file.unlink()
        print(f"Cache cleared: {cache_file}")
    else:
        print("No cache file found")


def get_cache_stats(cache_dir=None):
    """Get statistics about the cache

    Args:
        cache_dir: Directory containing cache file. If None, uses default.

    Returns:
        dict: Cache statistics including size, location, etc.
    """
    cache = get_cache(cache_dir)
    return {
        'size': cache.size(),
        'location': str(cache.cache_file),
        'exists': cache.cache_file.exists(),
    }


class BRDInChICache:
    """Persistent cache for InChI key to ChEMBL ID mappings"""

    def __init__(self, cache_dir=None):
        """Initialize cache

        Args:
            cache_dir: Directory to store cache file. If None, uses default.
        """
        if cache_dir is None:
            cache_dir = DEFAULT_BRD_CACHE_DIR

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'inchi_chembl_cache.pkl'
        self._cache = self._load_cache()

    def _load_cache(self):
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    if isinstance(cache, dict):
                        logger.info(f"Loaded {len(cache)} cached InChI-ChEMBL mappings")
                        return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            logger.info(f"Saved {len(self._cache)} mappings to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get(self, inchi_key):
        """Get ChEMBL ID for an InChI key from cache"""
        return self._cache.get(inchi_key)

    def set(self, inchi_key, chembl_id):
        """Set ChEMBL ID for an InChI key in cache"""
        self._cache[inchi_key] = chembl_id

    def __contains__(self, inchi_key):
        """Check if InChI key is in cache"""
        return inchi_key in self._cache

    def save(self):
        """Explicitly save cache to disk"""
        self._save_cache()

    def size(self):
        """Return number of entries in cache"""
        return len(self._cache)


def _query_chembl_single_inchi(inchi_key: str, cache: BRDInChICache) -> Optional[str]:
    """Query ChEMBL for a single InChI key

    Args:
        inchi_key: InChI key to query
        cache: BRDInChICache instance

    Returns:
        ChEMBL ID if found, None otherwise
    """
    if not inchi_key or pd.isna(inchi_key):
        return None

    if inchi_key in cache:
        return cache.get(inchi_key)

    try:
        from chembl_webresource_client.new_client import new_client
        molecule_api = new_client.molecule
        mols = molecule_api.filter(
            molecule_structures__standard_inchi_key=inchi_key
        ).only(['molecule_chembl_id'])

        if mols:
            chembl_id = mols[0]['molecule_chembl_id']
            cache.set(inchi_key, chembl_id)
            return chembl_id
    except Exception as e:
        logger.debug(f"Failed to map InChI key {inchi_key}: {e}")

    cache.set(inchi_key, None)
    return None


def brd_to_chembl_batch(brd_ids: Union[List[str], pd.Series],
                       metadata_df: Optional[pd.DataFrame] = None,
                       max_workers: int = 10,
                       cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Batch convert BRD IDs to ChEMBL IDs via InChI keys

    Maps BRD IDs (Broad Institute compound IDs from LINCS) to ChEMBL IDs
    by looking up InChI keys in LINCS metadata and querying ChEMBL API.
    Uses persistent caching to minimize API calls.

    Args:
        brd_ids: List or Series of BRD identifiers (e.g., 'BRD-K12345678-001-01-1')
        metadata_df: DataFrame with 'pert_id' and 'inchi_key' columns.
                    If None, will attempt to load from LINCS metadata file.
        max_workers: Number of parallel workers for ChEMBL queries
        cache_dir: Directory for cache file. If None, uses default.

    Returns:
        DataFrame with columns:
            - brd_id: Original BRD ID
            - chembl_id: Mapped ChEMBL ID (or None if not found)
            - confidence: Confidence score (0.95 if mapped, 0.0 otherwise)

    Example:
        >>> brd_ids = ['BRD-K12345678', 'BRD-K87654321']
        >>> results = brd_to_chembl_batch(brd_ids)
        >>> print(results[['brd_id', 'chembl_id']])
    """
    if isinstance(brd_ids, pd.Series):
        brd_ids = brd_ids.tolist()

    unique_brds = list(dict.fromkeys(brd_ids))
    logger.info(f"Processing {len(unique_brds)} unique BRD IDs")

    # Load metadata if not provided
    if metadata_df is None:
        data_dir = os.environ.get('CONTEXTPERT_DATA_DIR')
        if data_dir is None:
            raise ValueError("CONTEXTPERT_DATA_DIR environment variable not set and no metadata_df provided")

        # Try to find LINCS metadata file with InChI keys
        possible_paths = [
            os.path.join(data_dir, 'GSE92742_Broad_LINCS_pert_info.txt'),
            os.path.join(data_dir, 'GSE92742_Broad_LINCS_pert_info (1).txt'),
        ]

        metadata_path = None
        for path in possible_paths:
            if os.path.exists(path):
                metadata_path = path
                break

        if metadata_path is None:
            raise ValueError(
                "Could not find LINCS metadata file. "
                "Expected 'GSE92742_Broad_LINCS_pert_info.txt' in CONTEXTPERT_DATA_DIR. "
                "Alternatively, provide metadata_df with 'pert_id' and 'inchi_key' columns."
            )

        logger.info(f"Loading LINCS metadata from {metadata_path}")
        metadata_df = pd.read_csv(metadata_path, sep='\t')

    # Create mapping dataframe
    brd_df = pd.DataFrame({'brd_id': unique_brds})
    brd_df['core_brd'] = brd_df['brd_id'].str[:13]  # Extract core BRD ID

    # Merge with metadata to get InChI keys
    required_cols = ['pert_id', 'inchi_key']
    available_cols = [col for col in required_cols if col in metadata_df.columns]

    if 'inchi_key' not in available_cols:
        raise ValueError("metadata_df must contain 'inchi_key' column")

    merged = brd_df.merge(
        metadata_df[available_cols].drop_duplicates('pert_id'),
        left_on='core_brd',
        right_on='pert_id',
        how='left'
    )

    # Get unique InChI keys for querying
    unique_inchi_keys = merged['inchi_key'].dropna().unique().tolist()
    logger.info(f"Found {len(unique_inchi_keys)} unique InChI keys")

    # Initialize cache
    cache = BRDInChICache(cache_dir=cache_dir)

    # Query ChEMBL in parallel for uncached keys
    uncached_keys = [k for k in unique_inchi_keys if k not in cache]

    if uncached_keys:
        logger.info(f"Querying ChEMBL for {len(uncached_keys)} uncached InChI keys using {max_workers} workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(_query_chembl_single_inchi, key, cache): key
                for key in uncached_keys
            }
            for i, future in enumerate(concurrent.futures.as_completed(future_to_key)):
                if i % 100 == 0 and i > 0:
                    logger.info(f"Processed {i}/{len(uncached_keys)} InChI keys")
                try:
                    future.result()
                except Exception as e:
                    logger.debug(f"Error processing key: {e}")

        cache.save()
    else:
        logger.info("All InChI keys found in cache!")

    # Map InChI keys to ChEMBL IDs
    inchi_to_chembl = {k: cache.get(k) for k in unique_inchi_keys}
    merged['chembl_id'] = merged['inchi_key'].map(inchi_to_chembl)
    merged['confidence'] = merged['chembl_id'].notna().astype(float) * 0.95

    # Return results
    results = merged[['brd_id', 'chembl_id', 'confidence']].copy()
    return results
