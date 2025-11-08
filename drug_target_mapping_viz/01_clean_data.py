#!/usr/bin/env python
"""
Drug-Target Mapping Visualization using LINCS Expression Data

Visualizes drug-target interaction prediction using gene expression profiles from
LINCS L1000 data:
- Drug representations: Expression profiles from compound perturbations (trt_cp)
- Target representations: Expression profiles from shRNA knockdowns (trt_sh)

Both are aggregated by averaging across replicates for each unique perturbation.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']

print("=" * 80)
print("DRUG-TARGET EXPRESSION VISUALIZATION")
print("=" * 80)
print("\nThis example uses LINCS L1000 gene expression profiles to visualize")
print("drug-target interactions:")
print("  - Drugs: Expression from compound perturbations (trt_cp)")
print("  - Targets: Expression from shRNA knockdowns (trt_sh)")
print()


# ============================================================================
# Part 1: Load and Process Drug Data (trt_cp)
# ============================================================================
print("=" * 80)
print("LOADING DRUG DATA (COMPOUND PERTURBATIONS)")
print("=" * 80)

trt_cp_path = os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv')
print(f"\nLoading compound perturbation data from: {trt_cp_path}")
trt_cp_df = pd.read_csv(trt_cp_path)

print(f"Loaded trt_cp data:")
print(f"  Total samples: {len(trt_cp_df):,}")
print(f"  Unique BRD IDs: {trt_cp_df['pert_id'].nunique():,}")
print(f"  Unique canonical SMILES: {trt_cp_df['canonical_smiles'].nunique():,}")

# Read Entrez features from trt_sh_qc_gene_cols.txt
gene_cols_path = os.path.join(DATA_DIR, 'trt_sh_qc_gene_cols.txt')
print(f"\nLoading gene columns from: {gene_cols_path}")
with open(gene_cols_path, 'r') as f:
    gene_cols = [line.strip() for line in f]

print(f"  Loaded {len(gene_cols)} gene features (Entrez IDs)")
print(f"  Example Entrez IDs: {gene_cols[:5]}")

# Aggregate expression by SMILES (average across replicates)
print("\nAggregating expression profiles by SMILES...")
agg_dict = {col: 'mean' for col in gene_cols}
agg_dict['canonical_smiles'] = 'first'

drug_expr_df = (
    trt_cp_df.groupby('pert_id')[gene_cols + ['canonical_smiles']]
    .agg(agg_dict)
    .reset_index()
)

print(f"  Aggregated to {len(drug_expr_df):,} unique compounds")

# Prepare drug prediction dataframe
drug_preds = drug_expr_df[['canonical_smiles'] + gene_cols].rename(columns={'canonical_smiles': 'smiles'})

print(f"\nFinal drug representation:")
print(f"  Unique compounds: {len(drug_preds)}")
print(f"  Gene features (Entrez): {len(drug_preds.columns) - 1}")
print(f"  Shape: {drug_preds.shape}")


# ============================================================================
# Part 2: Load and Process Target Data (trt_sh)
# ============================================================================
print("\n" + "=" * 80)
print("LOADING TARGET DATA (shRNA KNOCKDOWNS)")
print("=" * 80)

trt_sh_genes_path = os.path.join(DATA_DIR, 'trt_sh_genes_qc.csv')
print(f"\nLoading shRNA knockdown data with target annotations from: {trt_sh_genes_path}")
trt_sh_df = pd.read_csv(trt_sh_genes_path, low_memory=False)

print(f"Loaded trt_sh_genes_qc data:")
print(f"  Total samples: {len(trt_sh_df):,}")
print(f"  Unique perturbation IDs: {trt_sh_df['pert_id'].nunique():,}")
print(f"  Samples with target annotation: {trt_sh_df['ensembl_id'].notna().sum():,}")
print(f"  Unique target genes: {trt_sh_df['ensembl_id'].nunique():,}")

# Use the same gene col features (loaded in TODO 1)
print(f"\n  Using same {len(gene_cols)} gene features for target data")

# Filter to only perturbations with target annotations
print("\nFiltering to perturbations with target annotations...")
trt_sh_df = trt_sh_df[trt_sh_df['ensembl_id'].notna()].copy()
print(f"  Retained samples: {len(trt_sh_df):,}")

# Aggregate expression by target gene (average across perturbations targeting same gene)
print("\nAggregating by target gene...")
agg_dict_sh = {col: 'mean' for col in gene_cols}

target_expr_df = (
    trt_sh_df.groupby('ensembl_id')[gene_cols]
    .mean()
    .reset_index()
)

print(f"  Aggregated to {len(target_expr_df):,} unique target genes")

# Prepare target prediction dataframe (targetId + same gene_cols as drug data)
target_preds = target_expr_df.rename(columns={'ensembl_id': 'targetId'})

print(f"\nFinal target representation:")
print(f"  Unique targets: {len(target_preds)}")
print(f"  Gene features (Entrez): {len(target_preds.columns) - 1}")
print(f"  Shape: {target_preds.shape}")


# ============================================================================
# Part 3: Filter to OpenTargets Drug-Target Pairs used in contextpert.submit_drug_target_mapping
# ============================================================================
print("\n" + "=" * 80)
print("FILTERING TO OPENTARGETS DRUG-TARGET PAIRS")
print("=" * 80)

# Load OpenTargets LINCS drug-target pairs
pairs_path = os.path.join(DATA_DIR, 'opentargets/drug_target_pairs_csv/drug_target_pairs_lincs.csv')
print(f"\nLoading OpenTargets LINCS pairs from: {pairs_path}")
drug_target_pairs = pd.read_csv(pairs_path)

print(f"Loaded OpenTargets pairs:")
print(f"  Total pairs: {len(drug_target_pairs):,}")
print(f"  Unique drugs (SMILES): {drug_target_pairs['smiles'].nunique():,}")
print(f"  Unique targets (Ensembl): {drug_target_pairs['targetId'].nunique():,}")

# Filter drug predictions to only those in OpenTargets
print("\nFiltering drugs to OpenTargets pairs...")
drug_preds_filtered = drug_preds[drug_preds['smiles'].isin(drug_target_pairs['smiles'])].copy()
print(f"  Drugs before filtering: {len(drug_preds):,}")
print(f"  Drugs after filtering: {len(drug_preds_filtered):,}")

# Filter target predictions to only those in OpenTargets
print("\nFiltering targets to OpenTargets pairs...")
target_preds_filtered = target_preds[target_preds['targetId'].isin(drug_target_pairs['targetId'])].copy()
print(f"  Targets before filtering: {len(target_preds):,}")
print(f"  Targets after filtering: {len(target_preds_filtered):,}")


# ============================================================================
# Part 4: Combine and create low-dim UMAP
# ============================================================================
print("\n" + "=" * 80)
print("COMBINING DATA AND CREATING UMAP EMBEDDING")
print("=" * 80)

from umap import UMAP

# Prepare drug data with modality label
print("\nPreparing combined dataset...")
drug_data = drug_preds_filtered.copy()
drug_data['modality'] = 'compound'
drug_data = drug_data.rename(columns={'smiles': 'id'})

# Prepare target data with modality label
target_data = target_preds_filtered.copy()
target_data['modality'] = 'shRNA'
target_data = target_data.rename(columns={'targetId': 'id'})

# Combine both datasets
combined_data = pd.concat([drug_data, target_data], axis=0, ignore_index=True)
print(f"  Combined dataset size: {len(combined_data):,}")
print(f"    Compounds: {(combined_data['modality'] == 'compound').sum():,}")
print(f"    shRNA targets: {(combined_data['modality'] == 'shRNA').sum():,}")

# Extract gene expression features
feature_cols = gene_cols
X = combined_data[feature_cols].values
print(f"\nFeature matrix shape: {X.shape}")
print(f"  Samples: {X.shape[0]}")
print(f"  Features (genes): {X.shape[1]}")

# Apply UMAP dimensionality reduction
print("\nApplying UMAP dimensionality reduction...")
print("  Parameters: n_neighbors=30, min_dist=0.1, n_components=2, random_state=42")
reducer = UMAP(random_state=42, n_neighbors=30, min_dist=0.1, n_components=2)
embedding = reducer.fit_transform(X)

print(f"  UMAP embedding shape: {embedding.shape}")
print(f"  UMAP complete!")


# ============================================================================
# Part 5: Create new plot df with id, name (symbol for gene, compound name for drug), pert_type, maps_to (list of ids), and UMAP coords.
# ============================================================================
print("\n" + "=" * 80)
print("CREATING PLOT DATAFRAME")
print("=" * 80)

import json

# Step 1: Create name mappings
print("\nCreating name mappings...")

# Map ensembl_id -> gene_symbol for targets
ensembl_to_symbol = dict(zip(trt_sh_df['ensembl_id'], trt_sh_df['gene_symbol']))
print(f"  Created gene symbol mapping for {len(ensembl_to_symbol):,} targets")

# Map smiles -> prefName for drugs
smiles_to_drugname = dict(zip(drug_target_pairs['smiles'], drug_target_pairs['prefName']))
print(f"  Created drug name mapping for {len(smiles_to_drugname):,} drugs")

# Step 2: Add UMAP coordinates to combined_data
print("\nAdding UMAP coordinates...")
combined_data['umap_1'] = embedding[:, 0]
combined_data['umap_2'] = embedding[:, 1]
print(f"  Added UMAP coordinates to {len(combined_data):,} samples")

# Step 3: Build maps_to connections
print("\nBuilding drug-target connections...")

# Create mapping dictionaries for efficient lookup
# For drugs: smiles -> list of targetIds
drug_to_targets = {}
for _, row in drug_target_pairs.iterrows():
    smiles = row['smiles']
    target = row['targetId']
    if smiles not in drug_to_targets:
        drug_to_targets[smiles] = []
    drug_to_targets[smiles].append(target)

# For targets: targetId -> list of smiles
target_to_drugs = {}
for _, row in drug_target_pairs.iterrows():
    smiles = row['smiles']
    target = row['targetId']
    if target not in target_to_drugs:
        target_to_drugs[target] = []
    target_to_drugs[target].append(smiles)

print(f"  Built connections for {len(drug_to_targets):,} drugs and {len(target_to_drugs):,} targets")

# Step 4: Create final plot DataFrame
print("\nCreating final plot DataFrame...")

# Add name column based on modality
combined_data['name'] = combined_data.apply(
    lambda row: smiles_to_drugname.get(row['id'], row['id']) if row['modality'] == 'compound'
    else ensembl_to_symbol.get(row['id'], row['id']),
    axis=1
)

# Add maps_to column as JSON strings
combined_data['maps_to'] = combined_data.apply(
    lambda row: json.dumps(drug_to_targets.get(row['id'], [])) if row['modality'] == 'compound'
    else json.dumps(target_to_drugs.get(row['id'], [])),
    axis=1
)

# Rename modality to pert_type for clarity
combined_data['pert_type'] = combined_data['modality']

# Select and reorder columns for final output
plot_df = combined_data[['id', 'name', 'pert_type', 'maps_to', 'umap_1', 'umap_2']].copy()

print(f"  Final DataFrame shape: {plot_df.shape}")
print(f"    Columns: {list(plot_df.columns)}")

# Step 5: Save to CSV
output_path = os.path.join('drug_target_mapping_viz', 'expr_plot_data.csv')
print(f"\nSaving plot data to: {output_path}")
plot_df.to_csv(output_path, index=False)
print(f"  ✓ Saved successfully!")

# Print summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total samples in plot: {len(plot_df):,}")
print(f"  Compounds: {(plot_df['pert_type'] == 'compound').sum():,}")
print(f"  shRNA targets: {(plot_df['pert_type'] == 'shRNA').sum():,}")
print()

# Count total connections
total_connections = sum(len(json.loads(maps_to)) for maps_to in plot_df['maps_to'])
print(f"Total drug-target connections: {total_connections:,}")

# Calculate connection statistics
drug_df = plot_df[plot_df['pert_type'] == 'compound'].copy()
target_df = plot_df[plot_df['pert_type'] == 'shRNA'].copy()

drug_df['n_targets'] = drug_df['maps_to'].apply(lambda x: len(json.loads(x)))
target_df['n_drugs'] = target_df['maps_to'].apply(lambda x: len(json.loads(x)))

print(f"\nDrug statistics:")
print(f"  Avg targets per drug: {drug_df['n_targets'].mean():.2f}")
print(f"  Max targets per drug: {drug_df['n_targets'].max()}")
print(f"  Drugs with no targets: {(drug_df['n_targets'] == 0).sum()}")

print(f"\nTarget statistics:")
print(f"  Avg drugs per target: {target_df['n_drugs'].mean():.2f}")
print(f"  Max drugs per target: {target_df['n_drugs'].max()}")
print(f"  Targets with no drugs: {(target_df['n_drugs'] == 0).sum()}")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)