#!/usr/bin/env python
"""
Filter Generic Disease Names from OpenTargets Data

Removes overly broad/generic disease classifications (e.g., "cancer", "neoplasm")
to focus evaluation on specific disease types. Ensures filtered disease-drug triples
maintain valid evaluation structure (2+ unique target signatures per disease).

Input:
    ${CONTEXTPERT_DATA_DIR}/opentargets/disease_drug_triples_csv/disease_drug_triples.csv
    ${CONTEXTPERT_DATA_DIR}/opentargets/disease_target_pairs_csv/disease_target_pairs.csv

Output:
    ${CONTEXTPERT_DATA_DIR}/opentargets/disease_drug_triples_csv/disease_drug_triples_filtered.csv
    ${CONTEXTPERT_DATA_DIR}/opentargets/disease_target_pairs_csv/disease_target_pairs_filtered.csv
"""

import os
import pandas as pd

TO_REMOVE = [
    'neoplasm',
    'cancer',
    'breast cancer',
    'breast neoplasm',
    'lymphoma',
    'colorectal neoplasm',
    'neuroendocrine neoplasm',
    'non-Hodgkins lymphoma',
]

print("=" * 80)
print("FILTERING GENERIC DISEASE NAMES")
print("=" * 80)

DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')

# Load data
print(f"\nLoading data from: {DATA_DIR}/opentargets/")
triples_df = pd.read_csv(os.path.join(DATA_DIR, 'opentargets/disease_drug_triples_csv/disease_drug_triples.csv'))
targets_df = pd.read_csv(os.path.join(DATA_DIR, 'opentargets/disease_target_pairs_csv/disease_target_pairs.csv'))

print(f"\nOriginal data:")
print(f"  Disease-drug triples: {len(triples_df):,}")
print(f"    Unique diseases: {triples_df['diseaseId'].nunique():,}")
print(f"    Unique drugs: {triples_df['drugId'].nunique():,}")
print(f"  Disease-target pairs: {len(targets_df):,}")
print(f"    Unique diseases: {targets_df['diseaseId'].nunique():,}")
print(f"    Unique targets: {targets_df['targetId'].nunique():,}")

# Filter out generic disease names (case-insensitive)
print(f"\nFiltering out generic disease names:")
print(f"  Removing: {TO_REMOVE}")

# Convert to lowercase for case-insensitive matching
to_remove_lower = [name.lower() for name in TO_REMOVE]

# Filter triples
triples_before = len(triples_df)
triples_df = triples_df[~triples_df['diseaseName'].str.lower().isin(to_remove_lower)].copy()
triples_removed = triples_before - len(triples_df)

# Filter targets
targets_before = len(targets_df)
targets_df = targets_df[~targets_df['diseaseName'].str.lower().isin(to_remove_lower)].copy()
targets_removed = targets_before - len(targets_df)

print(f"\nAfter generic name filtering:")
print(f"  Disease-drug triples: {len(triples_df):,} (removed {triples_removed:,})")
print(f"  Disease-target pairs: {len(targets_df):,} (removed {targets_removed:,})")

# Sanity check: Verify diseases still have 2+ unique target signatures
print(f"\nSanity check: Verifying diseases have 2+ unique target signatures...")
disease_sig_counts = triples_df.groupby('diseaseId')['targets'].nunique()
invalid_diseases = disease_sig_counts[disease_sig_counts < 2]

if len(invalid_diseases) > 0:
    print(f"  WARNING: {len(invalid_diseases)} diseases now have < 2 unique target signatures:")
    for disease_id, sig_count in invalid_diseases.items():
        disease_name = triples_df[triples_df['diseaseId'] == disease_id]['diseaseName'].iloc[0]
        print(f"    - {disease_id} ({disease_name}): {sig_count} signature(s)")

    print(f"\n  Removing these {len(invalid_diseases)} diseases from triples...")
    triples_df = triples_df[~triples_df['diseaseId'].isin(invalid_diseases.index)].copy()
    print(f"  Disease-drug triples after removal: {len(triples_df):,}")
else:
    print(f"  ✓ All diseases have 2+ unique target signatures")

print(f"\nFinal filtered data:")
print(f"  Disease-drug triples: {len(triples_df):,}")
print(f"    Unique diseases: {triples_df['diseaseId'].nunique():,}")
print(f"    Unique drugs: {triples_df['drugId'].nunique():,}")
print(f"    Unique target signatures: {triples_df['targets'].nunique():,}")
print(f"  Disease-target pairs: {len(targets_df):,}")
print(f"    Unique diseases: {targets_df['diseaseId'].nunique():,}")
print(f"    Unique targets: {targets_df['targetId'].nunique():,}")

# Generate summary tables
print("\n" + "=" * 80)
print("DISEASE-DRUG TRIPLES SUMMARY (Top 10)")
print("=" * 80)
triples_summary = triples_df.groupby(['diseaseId', 'diseaseName']).agg({
    'targets': 'nunique',  # unique target signatures
    'drugId': 'nunique'    # unique drugs
}).reset_index()
triples_summary.columns = ['diseaseId', 'diseaseName', 'num_target_signatures', 'num_drugs']
triples_summary = triples_summary.sort_values('num_drugs', ascending=False)

print(f"\n{triples_summary.head(10).to_string(index=False)}")
print(f"\nTotal diseases: {len(triples_summary)}")

print("\n" + "=" * 80)
print("DISEASE-TARGET PAIRS SUMMARY (Top 10)")
print("=" * 80)
targets_summary = targets_df.groupby(['diseaseId', 'diseaseName']).agg({
    'targetId': 'nunique'  # unique targets
}).reset_index()
targets_summary.columns = ['diseaseId', 'diseaseName', 'num_targets']
targets_summary = targets_summary.sort_values('num_targets', ascending=False)

print(f"\n{targets_summary.head(10).to_string(index=False)}")
print(f"\nTotal diseases: {len(targets_summary)}")

# Save filtered datasets
triples_output = os.path.join(DATA_DIR, 'opentargets/disease_drug_triples_csv/disease_drug_triples_filtered.csv')
targets_output = os.path.join(DATA_DIR, 'opentargets/disease_target_pairs_csv/disease_target_pairs_filtered.csv')

print(f"\nSaving filtered datasets:")
print(f"  {triples_output}")
triples_df.to_csv(triples_output, index=False)

print(f"  {targets_output}")
targets_df.to_csv(targets_output, index=False)

print("\n" + "=" * 80)
print("FILTERING COMPLETE")
print("=" * 80)
print(f"\nOutput files:")
print(f"  {triples_output}")
print(f"  {targets_output}")
print("\n✓ Done")