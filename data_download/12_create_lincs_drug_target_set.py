#!/usr/bin/env python
"""
Create LINCS-Filtered Drug-Target Set for Evaluation

Filters the OpenTargets drug-target pairs to only include drugs and targets
present in the quality-controlled LINCS datasets. Ensures the filtered dataset
maintains sufficient coverage for meaningful evaluation.

Input:
    ${CONTEXTPERT_DATA_DIR}/trt_cp_smiles_qc.csv
    ${CONTEXTPERT_DATA_DIR}/trt_sh_qc.csv
    ${CONTEXTPERT_DATA_DIR}/opentargets/drug_target_pairs_csv/drug_target_pairs.csv

Output:
    ${CONTEXTPERT_DATA_DIR}/opentargets/drug_target_pairs_csv/drug_target_pairs_lincs.csv
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
print("CREATING LINCS-FILTERED DRUG-TARGET SET")
print("=" * 80)

# Load LINCS compound data (drugs)
print("\n" + "=" * 80)
print("LOADING LINCS DRUG DATA")
print("=" * 80)

lincs_cp_path = os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv')
print(f"\nLoading LINCS compound data from: {lincs_cp_path}")
lincs_cp_df = pd.read_csv(lincs_cp_path)

print(f"Loaded LINCS compound data:")
print(f"  Total samples: {len(lincs_cp_df):,}")
print(f"  Unique BRD IDs: {lincs_cp_df['pert_id'].nunique():,}")
print(f"  Unique canonical SMILES: {lincs_cp_df['canonical_smiles'].nunique():,}")

# Get unique SMILES from LINCS
lincs_smiles_set = set(lincs_cp_df['canonical_smiles'].dropna().unique())
print(f"\nUnique LINCS drug SMILES: {len(lincs_smiles_set):,}")

# Load LINCS shRNA data (targets)
print("\n" + "=" * 80)
print("LOADING LINCS TARGET DATA")
print("=" * 80)

lincs_sh_path = os.path.join(DATA_DIR, 'trt_sh_qc.csv')
print(f"\nLoading LINCS shRNA data from: {lincs_sh_path}")
lincs_sh_df = pd.read_csv(lincs_sh_path, low_memory=False)

print(f"Loaded LINCS shRNA data:")
print(f"  Total samples: {len(lincs_sh_df):,}")
print(f"  Unique perturbation IDs: {lincs_sh_df['pert_id'].nunique():,}")

# Identify gene columns (ENSG IDs)
metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                 'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                 'pct_self_rank_q25']
gene_cols = [col for col in lincs_sh_df.columns if col not in metadata_cols]

print(f"  Gene columns (ENSG IDs): {len(gene_cols)}")
print(f"  Example ENSG IDs: {gene_cols[:10]}")

# Get unique target ENSG IDs from LINCS
lincs_target_set = set(gene_cols)
print(f"\nUnique LINCS target ENSG IDs: {len(lincs_target_set):,}")

# Load OpenTargets drug-target pairs
print("\n" + "=" * 80)
print("LOADING OPENTARGETS DRUG-TARGET PAIRS")
print("=" * 80)

opentargets_path = os.path.join(
    DATA_DIR,
    'opentargets/drug_target_pairs_csv/drug_target_pairs.csv'
)
print(f"\nLoading OpenTargets drug-target pairs from: {opentargets_path}")
opentargets_df = pd.read_csv(opentargets_path)

print(f"Loaded OpenTargets data:")
print(f"  Total pairs: {len(opentargets_df):,}")
print(f"  Unique drugs (SMILES): {opentargets_df['smiles'].nunique():,}")
print(f"  Unique targets (ENSG): {opentargets_df['targetId'].nunique():,}")

# Canonicalize OpenTargets SMILES for consistent comparison
print("\n" + "=" * 80)
print("CANONICALIZING OPENTARGETS SMILES")
print("=" * 80)

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

print(f"  Canonicalized {len(opentargets_df):,} pairs")

# Find overlaps
print("\n" + "=" * 80)
print("COMPUTING OVERLAPS")
print("=" * 80)

print(f"\nDrug (SMILES) overlap:")
opentargets_smiles_set = set(opentargets_df['smiles'].unique())
drug_overlap = lincs_smiles_set & opentargets_smiles_set

print(f"  LINCS drugs: {len(lincs_smiles_set):,}")
print(f"  OpenTargets drugs: {len(opentargets_smiles_set):,}")
print(f"  Overlapping drugs: {len(drug_overlap):,}")
print(f"  Overlap % (vs LINCS): {len(drug_overlap)/len(lincs_smiles_set)*100:.1f}%")
print(f"  Overlap % (vs OpenTargets): {len(drug_overlap)/len(opentargets_smiles_set)*100:.1f}%")

print(f"\nTarget (ENSG) overlap:")
opentargets_target_set = set(opentargets_df['targetId'].unique())
target_overlap = lincs_target_set & opentargets_target_set

print(f"  LINCS targets: {len(lincs_target_set):,}")
print(f"  OpenTargets targets: {len(opentargets_target_set):,}")
print(f"  Overlapping targets: {len(target_overlap):,}")
print(f"  Overlap % (vs LINCS): {len(target_overlap)/len(lincs_target_set)*100:.1f}%")
print(f"  Overlap % (vs OpenTargets): {len(target_overlap)/len(opentargets_target_set)*100:.1f}%")

# Filter OpenTargets to overlapping drugs and targets
print("\n" + "=" * 80)
print("FILTERING TO LINCS OVERLAP")
print("=" * 80)

print(f"\nFiltering OpenTargets to overlapping drugs and targets...")
filtered_df = opentargets_df[
    opentargets_df['smiles'].isin(drug_overlap) &
    opentargets_df['targetId'].isin(target_overlap)
].copy()

print(f"Filtered OpenTargets data:")
print(f"  Total pairs: {len(filtered_df):,}")
print(f"  Unique drugs (SMILES): {filtered_df['smiles'].nunique():,}")
print(f"  Unique targets (ENSG): {filtered_df['targetId'].nunique():,}")

# Calculate retention rates
print(f"\nRetention rates:")
print(f"  Pairs: {len(filtered_df)/len(opentargets_df)*100:.1f}%")
print(f"  Drugs: {filtered_df['smiles'].nunique()/opentargets_df['smiles'].nunique()*100:.1f}%")
print(f"  Targets: {filtered_df['targetId'].nunique()/opentargets_df['targetId'].nunique()*100:.1f}%")

# Validate minimum coverage
MIN_PAIRS = 10
MIN_DRUGS = 5
MIN_TARGETS = 5

print(f"\nValidating minimum coverage...")
print(f"  Minimum pairs required: {MIN_PAIRS}")
print(f"  Minimum drugs required: {MIN_DRUGS}")
print(f"  Minimum targets required: {MIN_TARGETS}")

if len(filtered_df) < MIN_PAIRS:
    print(f"\n  ✗ ERROR: Only {len(filtered_df)} pairs remaining (< {MIN_PAIRS})")
    print(f"    Insufficient data for meaningful evaluation")
    sys.exit(1)

if filtered_df['smiles'].nunique() < MIN_DRUGS:
    print(f"\n  ✗ ERROR: Only {filtered_df['smiles'].nunique()} drugs remaining (< {MIN_DRUGS})")
    print(f"    Insufficient data for meaningful evaluation")
    sys.exit(1)

if filtered_df['targetId'].nunique() < MIN_TARGETS:
    print(f"\n  ✗ ERROR: Only {filtered_df['targetId'].nunique()} targets remaining (< {MIN_TARGETS})")
    print(f"    Insufficient data for meaningful evaluation")
    sys.exit(1)

print(f"\n  ✓ All minimum coverage requirements met")

# Save filtered dataset
output_path = os.path.join(
    DATA_DIR,
    'opentargets/drug_target_pairs_csv/drug_target_pairs_lincs.csv'
)

print(f"\n" + "=" * 80)
print("SAVING FILTERED DATASET")
print("=" * 80)

print(f"\nSaving LINCS-filtered drug-target pairs to: {output_path}")
filtered_df.to_csv(output_path, index=False)

# Final summary
print("\n" + "=" * 80)
print("LINCS DRUG-TARGET SET CREATION COMPLETE")
print("=" * 80)

print(f"\nInput sources:")
print(f"  LINCS drugs: {lincs_cp_path}")
print(f"    Unique SMILES: {len(lincs_smiles_set):,}")
print(f"  LINCS targets: {lincs_sh_path}")
print(f"    Unique ENSG IDs: {len(lincs_target_set):,}")
print(f"  OpenTargets pairs: {opentargets_path}")
print(f"    Total pairs: {len(opentargets_df):,}")

print(f"\nOutput: {output_path}")
print(f"  Total pairs: {len(filtered_df):,}")
print(f"  Unique drugs: {filtered_df['smiles'].nunique():,}")
print(f"  Unique targets: {filtered_df['targetId'].nunique():,}")

print(f"\nOverlap summary:")
print(f"  Drugs in LINCS: {len(lincs_smiles_set):,}")
print(f"  Targets in LINCS: {len(lincs_target_set):,}")
print(f"  Drugs in OpenTargets (full): {len(opentargets_smiles_set):,}")
print(f"  Targets in OpenTargets (full): {len(opentargets_target_set):,}")
print(f"  Drugs in overlap: {len(drug_overlap):,}")
print(f"  Targets in overlap: {len(target_overlap):,}")
print(f"  Pairs in LINCS set: {len(filtered_df):,}")

print("\n✓ Done")
