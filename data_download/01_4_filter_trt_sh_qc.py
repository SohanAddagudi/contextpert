#!/usr/bin/env python
"""
Filter trt_sh with Quality Control Metrics

Applies the standard LINCS quality control filter to shRNA knockdown data.
Gene columns remain as Entrez IDs (matching trt_cp_smiles_qc.csv format).

Input:
    ${CONTEXTPERT_DATA_DIR}/trt_sh.csv

Output:
    ${CONTEXTPERT_DATA_DIR}/trt_sh_qc.csv
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')

print("=" * 80)
print("FILTERING trt_sh WITH QUALITY CONTROL")
print("=" * 80)

# Load trt_sh data
input_path = os.path.join(DATA_DIR, 'trt_sh.csv')
print(f"\nLoading shRNA perturbation data from: {input_path}")
df = pd.read_csv(input_path, low_memory=False)

print(f"Loaded trt_sh:")
print(f"  Total samples: {len(df):,}")
print(f"  Unique perturbation IDs: {df['pert_id'].nunique():,}")
print(f"  Shape: {df.shape}")

# Apply quality control filter
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

# Final statistics
print("\n" + "=" * 80)
print("FINAL STATISTICS")
print("=" * 80)

print(f"\nOriginal data:")
print(f"  Samples: {len(df):,}")
print(f"  Unique perturbation IDs: {df['pert_id'].nunique():,}")

print(f"\nFiltered data:")
print(f"  Samples: {len(df_filtered):,}")
print(f"  Unique perturbation IDs: {df_filtered['pert_id'].nunique():,}")

print(f"\nOverall removal:")
print(f"  Total removed: {len(df) - len(df_filtered):,} samples")
print(f"  Retention rate: {len(df_filtered)/len(df)*100:.1f}%")

# Save filtered data
output_path = os.path.join(DATA_DIR, 'trt_sh_qc.csv')
print(f"\nSaving filtered data to: {output_path}")
df_filtered.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("FILTERING COMPLETE")
print("=" * 80)
print(f"\nOutput: {output_path}")
print(f"  Filtered samples: {len(df_filtered):,}")
print(f"  Unique perturbation IDs: {df_filtered['pert_id'].nunique():,}")
print("\n✓ Done")
