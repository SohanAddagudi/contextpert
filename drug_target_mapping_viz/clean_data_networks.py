#!/usr/bin/env python
"""
Drug-Target Mapping Visualization using Network Model Betas

Visualizes drug-target interaction prediction using learned network model parameters:
- Drug representations: Beta matrices from ContextualizedCorrelation models trained on compound perturbations
- Target representations: Beta matrices from ContextualizedCorrelation models trained on shRNA knockdowns

Beta matrices are flattened and aggregated by averaging across replicates for each unique perturbation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']

print("=" * 80)
print("DRUG-TARGET NETWORK VISUALIZATION")
print("=" * 80)
print("\nThis example uses learned network model betas to visualize")
print("drug-target interactions:")
print("  - Drugs: Beta matrices from compound perturbation models")
print("  - Targets: Beta matrices from shRNA knockdown models")
print()


# ============================================================================
# Part 1: Load and Process Drug Data (trt_cp) - Network Betas
# ============================================================================
print("=" * 80)
print("LOADING DRUG DATA (COMPOUND PERTURBATIONS - NETWORK BETAS)")
print("=" * 80)

# Load beta arrays from network model
cp_beta_path = os.path.join(DATA_DIR, 'drug_target_networks/fingerprint_trt_cp_preds_drug_target/full_dataset_betas.npy')
print(f"\nLoading compound beta arrays from: {cp_beta_path}")
cp_betas = np.load(cp_beta_path)

print(f"Loaded beta arrays:")
print(f"  Shape: {cp_betas.shape}")
n_samples, n_pcs, _ = cp_betas.shape
print(f"  Total samples: {n_samples:,}")
print(f"  Beta matrix dimensions: {n_pcs} x {n_pcs}")

# Flatten beta matrices into feature vectors
print(f"\nFlattening beta matrices...")
n_features = n_pcs * n_pcs
feature_cols = [f'beta_{i}' for i in range(n_features)]
cp_beta_df = pd.DataFrame(cp_betas.reshape(n_samples, n_features), columns=feature_cols)
print(f"  Flattened to {n_features:,} features per sample")

# Load metadata to get inst_id
cp_meta_path = os.path.join(DATA_DIR, 'drug_target_networks/fingerprint_trt_cp_preds_drug_target/full_dataset_predictions.csv')
print(f"\nLoading metadata from: {cp_meta_path}")
cp_meta_df = pd.read_csv(cp_meta_path, usecols=['inst_id'])

# Combine betas with metadata
print(f"\nCombining betas with metadata...")
cp_model_data_df = pd.concat([cp_meta_df[['inst_id']], cp_beta_df], axis=1)
print(f"  Combined shape: {cp_model_data_df.shape}")

# Load LINCS metadata to get pert_id and SMILES
trt_cp_path = os.path.join(DATA_DIR, 'trt_cp_smiles_qc.csv')
print(f"\nLoading LINCS metadata from: {trt_cp_path}")
lincs_meta_df = pd.read_csv(trt_cp_path, usecols=['inst_id', 'pert_id', 'canonical_smiles'])

# Merge betas with LINCS metadata on inst_id
print(f"\nMerging with LINCS metadata...")
merged_df = pd.merge(cp_model_data_df, lincs_meta_df, on='inst_id', how='left')
print(f"  Merged shape: {merged_df.shape}")
print(f"  Unique BRD IDs: {merged_df['pert_id'].nunique():,}")

# Filter out invalid SMILES and canonicalize
from contextpert.utils import canonicalize_smiles

def safe_canonicalize(smiles):
    try:
        return canonicalize_smiles(smiles)
    except:
        return None

print(f"\nCanonicalizing SMILES...")
merged_df = merged_df[~merged_df['canonical_smiles'].isin(['-666', 'restricted'])]
merged_df = merged_df[merged_df['canonical_smiles'].notna()]
merged_df['smiles'] = merged_df['canonical_smiles'].apply(safe_canonicalize)
merged_df = merged_df[merged_df['smiles'].notna()]
print(f"  Samples after SMILES filtering: {len(merged_df):,}")

# Aggregate beta features by pert_id (average across replicates)
print("\nAggregating beta profiles by pert_id...")
agg_dict = {col: 'mean' for col in feature_cols}
agg_dict['smiles'] = 'first'

drug_beta_df = (
    merged_df.groupby('pert_id')[feature_cols + ['smiles']]
    .agg(agg_dict)
    .reset_index()
)

print(f"  Aggregated to {len(drug_beta_df):,} unique compounds")

# Prepare drug prediction dataframe
drug_preds = drug_beta_df[['smiles'] + feature_cols]

print(f"\nFinal drug representation:")
print(f"  Unique compounds: {len(drug_preds)}")
print(f"  Beta features: {len(drug_preds.columns) - 1:,}")
print(f"  Shape: {drug_preds.shape}")


# ============================================================================
# Part 2: Load and Process Target Data (trt_sh) - Network Betas
# ============================================================================
print("\n" + "=" * 80)
print("LOADING TARGET DATA (shRNA KNOCKDOWNS - NETWORK BETAS)")
print("=" * 80)

# Load beta arrays from network model for targets
sh_beta_path = os.path.join(DATA_DIR, 'drug_target_networks/aidoprot_trt_sh_preds_drug_target/full_dataset_betas.npy')
print(f"\nLoading shRNA beta arrays from: {sh_beta_path}")
sh_betas = np.load(sh_beta_path)

print(f"Loaded beta arrays:")
print(f"  Shape: {sh_betas.shape}")
sh_n_samples, sh_n_pcs, _ = sh_betas.shape
print(f"  Total samples: {sh_n_samples:,}")
print(f"  Beta matrix dimensions: {sh_n_pcs} x {sh_n_pcs}")

# Flatten beta matrices into feature vectors
print(f"\nFlattening beta matrices...")
sh_n_features = sh_n_pcs * sh_n_pcs
sh_feature_cols = [f'beta_{i}' for i in range(sh_n_features)]
sh_beta_df = pd.DataFrame(sh_betas.reshape(sh_n_samples, sh_n_features), columns=sh_feature_cols)
print(f"  Flattened to {sh_n_features:,} features per sample")

# Load metadata to get inst_id
sh_meta_path = os.path.join(DATA_DIR, 'drug_target_networks/aidoprot_trt_sh_preds_drug_target/full_dataset_predictions.csv')
print(f"\nLoading metadata from: {sh_meta_path}")
sh_meta_df = pd.read_csv(sh_meta_path, usecols=['inst_id'])

# Combine betas with metadata
print(f"\nCombining betas with metadata...")
sh_model_data_df = pd.concat([sh_meta_df[['inst_id']], sh_beta_df], axis=1)
print(f"  Combined shape: {sh_model_data_df.shape}")

# Load LINCS metadata to get ensembl_id and gene_symbol
trt_sh_genes_path = os.path.join(DATA_DIR, 'trt_sh_genes_qc.csv')
print(f"\nLoading LINCS shRNA metadata from: {trt_sh_genes_path}")
trt_sh_df = pd.read_csv(trt_sh_genes_path, usecols=['inst_id', 'ensembl_id', 'gene_symbol'], low_memory=False)

# Merge betas with LINCS metadata on inst_id
print(f"\nMerging with LINCS metadata...")
sh_merged_df = pd.merge(sh_model_data_df, trt_sh_df, on='inst_id', how='left')
print(f"  Merged shape: {sh_merged_df.shape}")

# Filter to only perturbations with target annotations
print("\nFiltering to perturbations with target annotations...")
sh_merged_df = sh_merged_df[sh_merged_df['ensembl_id'].notna()].copy()
print(f"  Retained samples: {len(sh_merged_df):,}")
print(f"  Unique target genes: {sh_merged_df['ensembl_id'].nunique():,}")

# Verify feature columns match between drugs and targets
if sh_n_features != n_features:
    print(f"\nWARNING: Feature dimensions differ!")
    print(f"  Drug features: {n_features:,}")
    print(f"  Target features: {sh_n_features:,}")
    print(f"  Adjusting to use minimum feature set...")
    min_features = min(n_features, sh_n_features)
    feature_cols = [f'beta_{i}' for i in range(min_features)]
    sh_feature_cols = feature_cols
else:
    print(f"\n  Using same {len(feature_cols):,} beta features for target data")
    sh_feature_cols = feature_cols

# Aggregate beta features by target gene (average across perturbations targeting same gene)
print("\nAggregating by target gene...")
agg_dict_sh = {col: 'mean' for col in sh_feature_cols}

target_beta_df = (
    sh_merged_df.groupby('ensembl_id')[sh_feature_cols]
    .mean()
    .reset_index()
)

print(f"  Aggregated to {len(target_beta_df):,} unique target genes")

# Prepare target prediction dataframe (targetId + same feature_cols as drug data)
target_preds = target_beta_df.rename(columns={'ensembl_id': 'targetId'})

print(f"\nFinal target representation:")
print(f"  Unique targets: {len(target_preds)}")
print(f"  Beta features: {len(target_preds.columns) - 1:,}")
print(f"  Shape: {target_preds.shape}")

# Clean up large arrays to free memory
del cp_betas, sh_betas, cp_beta_df, sh_beta_df
print(f"\n  Freed beta array memory")


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

# Extract beta features (already defined from drug data loading)
# feature_cols is already defined from Part 1
X = combined_data[feature_cols].values
print(f"\nFeature matrix shape: {X.shape}")
print(f"  Samples: {X.shape[0]}")
print(f"  Features (betas): {X.shape[1]:,}")

# Remove first 2 PCs of batch effects
print("\nRemoving first few principal components to reduce batch effects...")
pca = PCA(n_components=3)
pca.fit(X)

# Project data onto first PCs
first_pcs_projection = pca.transform(X)

# Reconstruct first PCs contribution
first_pcs_contribution = first_pcs_projection @ pca.components_

# Remove first PCs from data
X_corrected = X - first_pcs_contribution

print(f"  Total explained variance removed: {pca.explained_variance_ratio_.sum():.4f}")
print(f"  Corrected feature matrix shape: {X_corrected.shape}")

# Use corrected data for UMAP
X = X_corrected

# Apply UMAP dimensionality reduction
print("\nApplying UMAP dimensionality reduction...")
print("  Parameters: n_neighbors=30, min_dist=0.1, n_components=2, random_state=42")
reducer = UMAP(random_state=42, n_neighbors=50, min_dist=0.5, n_components=2)
embedding = reducer.fit_transform(X)

# reducer = PCA(n_components=2)
# embedding = reducer.fit_transform(X)

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
output_path = 'networks_plot_data.csv'
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