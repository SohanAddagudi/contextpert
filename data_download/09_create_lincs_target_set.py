#!/usr/bin/env python
"""
Create LINCS-filtered Target Set for Drug-Disease Evaluation

Filters the OpenTargets disease-drug triples to only include drugs that are
present in the LINCS dataset. Ensures the filtered dataset maintains valid
evaluation structure (all diseases have 2+ unique target signatures).

Input:
    ${CONTEXTPERT_DATA_DIR}/trt_cp_smiles.csv
    ${CONTEXTPERT_DATA_DIR}/opentargets/disease_drug_triples_csv/disease_drug_triples.csv

Output:
    ${CONTEXTPERT_DATA_DIR}/opentargets/disease_drug_triples_csv/disease_drug_triples_lincs.csv
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from contextpert.utils import canonicalize_smiles

DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')

print("=" * 80)
print("CREATING LINCS-FILTERED TARGET SET")
print("=" * 80)

# Load LINCS data with canonical SMILES
lincs_path = os.path.join(DATA_DIR, 'trt_cp_smiles.csv')
print(f"\nLoading LINCS data from: {lincs_path}")
lincs_df = pd.read_csv(lincs_path)

# Filter out bad SMILES
bad_smiles = ['-666', 'restricted']
lincs_df = lincs_df[~lincs_df['canonical_smiles'].isin(bad_smiles)].copy()
lincs_df = lincs_df[lincs_df['canonical_smiles'].notna()].copy()

print(f"Loaded LINCS data:")
print(f"  Total samples: {len(lincs_df):,}")
print(f"  Unique BRD IDs: {lincs_df['pert_id'].nunique():,}")

# Canonicalize SMILES for consistent comparison
print(f"\nCanonicalizing LINCS SMILES...")
failed_canonicalization = []

def safe_canonicalize(smiles):
    """Safely canonicalize SMILES, returning None on failure"""
    try:
        return canonicalize_smiles(smiles)
    except Exception:
        failed_canonicalization.append(smiles)
        return None

lincs_df['canonical_smiles_clean'] = lincs_df['canonical_smiles'].apply(safe_canonicalize)
lincs_df = lincs_df[lincs_df['canonical_smiles_clean'].notna()].copy()

if failed_canonicalization:
    print(f"  Warning: {len(failed_canonicalization)} SMILES failed canonicalization")

# Get unique LINCS SMILES (now canonicalized)
lincs_smiles_set = set(lincs_df['canonical_smiles_clean'].dropna().unique())
print(f"Loaded LINCS data:")
print(f"  Total samples: {len(lincs_df):,}")
print(f"  Unique canonical SMILES: {len(lincs_smiles_set):,}")

# Load OpenTargets disease-drug triples
opentargets_path = os.path.join(
    DATA_DIR,
    'opentargets/disease_drug_triples_csv/disease_drug_triples.csv'
)
print(f"\nLoading OpenTargets disease-drug triples from: {opentargets_path}")
opentargets_df = pd.read_csv(opentargets_path)

print(f"Loaded OpenTargets data:")
print(f"  Total triples: {len(opentargets_df):,}")
print(f"  Unique diseases: {opentargets_df['diseaseId'].nunique():,}")
print(f"  Unique drugs (SMILES): {opentargets_df['smiles'].nunique():,}")
print(f"  Unique target signatures: {opentargets_df['targets'].nunique():,}")

# Canonicalize OpenTargets SMILES for consistent comparison
print(f"\nCanonicalizing OpenTargets SMILES...")
failed_canonicalization = []

def safe_canonicalize(smiles):
    """Safely canonicalize SMILES, returning None on failure"""
    try:
        return canonicalize_smiles(smiles)
    except Exception:
        failed_canonicalization.append(smiles)
        return None

opentargets_df['smiles_canonical'] = opentargets_df['smiles'].apply(safe_canonicalize)

if failed_canonicalization:
    print(f"  Warning: {len(failed_canonicalization)} SMILES failed canonicalization")
    opentargets_df = opentargets_df[opentargets_df['smiles_canonical'].notna()].copy()

# Replace original SMILES with canonical version
opentargets_df['smiles'] = opentargets_df['smiles_canonical']
opentargets_df = opentargets_df.drop(columns=['smiles_canonical'])

# Find intersection of SMILES
print(f"\nFinding overlap between LINCS and OpenTargets...")
opentargets_smiles_set = set(opentargets_df['smiles'].unique())
overlap_smiles = lincs_smiles_set & opentargets_smiles_set

print(f"  LINCS unique SMILES: {len(lincs_smiles_set):,}")
print(f"  OpenTargets unique SMILES: {len(opentargets_smiles_set):,}")
print(f"  Overlapping SMILES: {len(overlap_smiles):,}")
print(f"  Overlap percentage (vs LINCS): {len(overlap_smiles)/len(lincs_smiles_set)*100:.1f}%")
print(f"  Overlap percentage (vs OpenTargets): {len(overlap_smiles)/len(opentargets_smiles_set)*100:.1f}%")

# Filter OpenTargets to only overlapping SMILES
print(f"\nFiltering OpenTargets to overlapping drugs...")
filtered_df = opentargets_df[opentargets_df['smiles'].isin(overlap_smiles)].copy()

print(f"Filtered OpenTargets data:")
print(f"  Total triples: {len(filtered_df):,}")
print(f"  Unique diseases: {filtered_df['diseaseId'].nunique():,}")
print(f"  Unique drugs (SMILES): {filtered_df['smiles'].nunique():,}")
print(f"  Unique target signatures: {filtered_df['targets'].nunique():,}")

# Validate: ensure all diseases still have 2+ unique target signatures
print(f"\nValidating disease-target signature counts...")
disease_sig_counts = filtered_df.groupby('diseaseId')['targets'].nunique()
invalid_diseases = disease_sig_counts[disease_sig_counts < 2]

if len(invalid_diseases) > 0:
    print(f"  Found {len(invalid_diseases)} diseases with < 2 unique target signatures")
    print(f"  Removing these diseases to maintain valid evaluation structure...")

    # Remove invalid diseases
    filtered_df = filtered_df[~filtered_df['diseaseId'].isin(invalid_diseases.index)].copy()

    print(f"\nFinal filtered data after removing invalid diseases:")
    print(f"  Total triples: {len(filtered_df):,}")
    print(f"  Unique diseases: {filtered_df['diseaseId'].nunique():,}")
    print(f"  Unique drugs (SMILES): {filtered_df['smiles'].nunique():,}")
    print(f"  Unique target signatures: {filtered_df['targets'].nunique():,}")
else:
    print(f"  ✓ All diseases have 2+ unique target signatures")

# Verify final counts
final_disease_sig_counts = filtered_df.groupby('diseaseId')['targets'].nunique()
assert final_disease_sig_counts.min() >= 2, "Invalid diseases remain in filtered dataset"

# Save the filtered dataset
output_path = os.path.join(
    DATA_DIR,
    'opentargets/disease_drug_triples_csv/disease_drug_triples_lincs.csv'
)
print(f"\nSaving LINCS-filtered target set to: {output_path}")
filtered_df.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("LINCS TARGET SET CREATION COMPLETE")
print("=" * 80)
print(f"\nOutput: {output_path}")
print(f"  Total triples: {len(filtered_df):,}")
print(f"  Unique diseases: {filtered_df['diseaseId'].nunique():,}")
print(f"  Unique drugs: {filtered_df['smiles'].nunique():,}")
print(f"  Unique target signatures: {filtered_df['targets'].nunique():,}")
print(f"\nOverlap summary:")
print(f"  Drugs in LINCS: {len(lincs_smiles_set):,}")
print(f"  Drugs in OpenTargets (full): {len(opentargets_smiles_set):,}")
print(f"  Drugs in overlap: {len(overlap_smiles):,}")
print(f"  Drugs in final LINCS set: {filtered_df['smiles'].nunique():,}")
print("\n✓ Done")
