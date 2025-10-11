"""
ChEMBL utilities for batch conversion of ChEMBL IDs to SMILES with caching
"""
import os
import pickle
import warnings
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem


# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.chembl_cache')


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
