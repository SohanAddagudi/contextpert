#!/usr/bin/env python
"""
Filter trt_cp_smiles with Quality Control Metrics

Applies the standard LINCS quality control filter (from prepare_train.ipynb) to
compound perturbation data, canonicalizes SMILES, and removes invalid entries.

Input:
    ${CONTEXTPERT_DATA_DIR}/trt_cp_smiles.csv

Output:
    ${CONTEXTPERT_DATA_DIR}/trt_cp_smiles_qc.csv
    Columns: All original columns with canonical_smiles updated
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from contextpert.utils import canonicalize_smiles

DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')

print("=" * 80)
print("FILTERING trt_cp_smiles WITH QUALITY CONTROL")
print("=" * 80)

# Load trt_cp_smiles data
input_path = os.path.join(DATA_DIR, 'trt_cp_smiles.csv')
print(f"\nLoading compound perturbation data from: {input_path}")
df = pd.read_csv(input_path, low_memory=False)

print(f"Loaded trt_cp_smiles:")
print(f"  Total samples: {len(df):,}")
print(f"  Unique BRD IDs: {df['pert_id'].nunique():,}")
print(f"  Shape: {df.shape}")

# Apply quality control filter (same as prepare_train.ipynb)
print("\n" + "=" * 80)
print("APPLYING QUALITY CONTROL FILTER")
print("=" * 80)

print("\nFiltering by quality control metrics...")
print("Removing samples where:")
print("  - distil_cc_q75 < 0.2 OR == -666 OR is NaN")
print("  - pct_self_rank_q25 > 5 OR == -666 OR is NaN")

condition = (
    (df['distil_cc_q75'] < 0.2) |
    (df['distil_cc_q75'] == -666) |
    (df['distil_cc_q75'].isna()) |
    (df['pct_self_rank_q25'] > 5) |
    (df['pct_self_rank_q25'] == -666) |
    (df['pct_self_rank_q25'].isna())
)

n_removed = condition.sum()
df_filtered = df[~condition].copy()

print(f"\nQuality control results:")
print(f"  Removed: {n_removed:,} low-quality samples")
print(f"  Remaining: {len(df_filtered):,} samples")
print(f"  Removal rate: {n_removed/len(df)*100:.1f}%")

# Filter out bad SMILES values before canonicalization
print("\n" + "=" * 80)
print("FILTERING INVALID SMILES")
print("=" * 80)

bad_smiles = ['-666', 'restricted']
n_before = len(df_filtered)

print(f"\nRemoving bad SMILES values: {bad_smiles}")
df_filtered = df_filtered[~df_filtered['canonical_smiles'].isin(bad_smiles)].copy()
print(f"  Removed {n_before - len(df_filtered):,} samples with bad SMILES")

print(f"\nRemoving NaN SMILES...")
n_before = len(df_filtered)
df_filtered = df_filtered[df_filtered['canonical_smiles'].notna()].copy()
print(f"  Removed {n_before - len(df_filtered):,} samples with NaN SMILES")

print(f"\nAfter filtering invalid SMILES: {len(df_filtered):,} samples")

# Canonicalize SMILES
print("\n" + "=" * 80)
print("CANONICALIZING SMILES")
print("=" * 80)

print(f"\nCanonicalizing SMILES for consistency...")
failed_canonicalization = []

def safe_canonicalize(smiles):
    """Safely canonicalize SMILES, returning None on failure"""
    try:
        return canonicalize_smiles(smiles)
    except Exception:
        failed_canonicalization.append(smiles)
        return None

df_filtered['canonical_smiles_clean'] = df_filtered['canonical_smiles'].apply(safe_canonicalize)

if failed_canonicalization:
    print(f"  Warning: {len(failed_canonicalization)} SMILES failed canonicalization")
    print(f"  Examples: {failed_canonicalization[:5]}")

# Remove rows where canonicalization failed
n_before = len(df_filtered)
df_filtered = df_filtered[df_filtered['canonical_smiles_clean'].notna()].copy()
n_failed = n_before - len(df_filtered)

if n_failed > 0:
    print(f"  Removed {n_failed:,} samples with failed canonicalization")

# Replace canonical_smiles with the cleaned version
df_filtered['canonical_smiles'] = df_filtered['canonical_smiles_clean']
df_filtered = df_filtered.drop(columns=['canonical_smiles_clean'])

print(f"\nCanonalization complete:")
print(f"  Successfully canonicalized: {len(df_filtered):,} samples")

# Final statistics
print("\n" + "=" * 80)
print("FINAL STATISTICS")
print("=" * 80)

print(f"\nOriginal data:")
print(f"  Samples: {len(df):,}")
print(f"  Unique BRD IDs: {df['pert_id'].nunique():,}")
print(f"  Unique SMILES: {df['canonical_smiles'].nunique():,}")

print(f"\nFiltered data:")
print(f"  Samples: {len(df_filtered):,}")
print(f"  Unique BRD IDs: {df_filtered['pert_id'].nunique():,}")
print(f"  Unique canonical SMILES: {df_filtered['canonical_smiles'].nunique():,}")

print(f"\nOverall removal:")
print(f"  Total removed: {len(df) - len(df_filtered):,} samples")
print(f"  Retention rate: {len(df_filtered)/len(df)*100:.1f}%")

# Save filtered data
output_path = os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv')
print(f"\nSaving filtered data to: {output_path}")
df_filtered.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("FILTERING COMPLETE")
print("=" * 80)
print(f"\nOutput: {output_path}")
print(f"  Filtered samples: {len(df_filtered):,}")
print(f"  Unique BRD IDs: {df_filtered['pert_id'].nunique():,}")
print(f"  Unique canonical SMILES: {df_filtered['canonical_smiles'].nunique():,}")
print("\n✓ Done")
