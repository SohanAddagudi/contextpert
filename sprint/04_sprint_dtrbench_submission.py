import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from contextpert import submit_drug_target_mapping
from contextpert.utils import canonicalize_smiles

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']
SPRINT_DIR = os.path.join(DATA_DIR, 'sprint')

print("=" * 80)
print("SPRINT DTR-BENCH EVALUATION")
print("=" * 80)
print("\nThis evaluates SPRINT drug and target embeddings on the DTR-Bench")
print("(drug-target retrieval) benchmark.\n")

# ============================================================================
# Load SPRINT Drug Embeddings
# ============================================================================
print("=" * 80)
print("LOADING DRUG DATA")
print("=" * 80)

drugs_df = pd.read_csv(os.path.join(SPRINT_DIR, 'drugs.csv'))
drug_embeddings = np.load(os.path.join(SPRINT_DIR, 'drug_embeddings.npy'))

print(f"Loaded drugs.csv: {len(drugs_df)} drugs")
print(f"Loaded drug_embeddings.npy: shape {drug_embeddings.shape}")

assert len(drugs_df) == drug_embeddings.shape[0], "Drug count mismatch!"

# Canonicalize SMILES
print("\nCanonicalizing SMILES...")
failed_canon = []

def safe_canonicalize(smiles):
    try:
        return canonicalize_smiles(smiles)
    except:
        failed_canon.append(smiles)
        return None

drugs_df['smiles_canonical'] = drugs_df['SMILES'].apply(safe_canonicalize)

if failed_canon:
    print(f"  Warning: {len(failed_canon)} SMILES failed canonicalization")

# Filter valid drugs
valid_drug_mask = drugs_df['smiles_canonical'].notna()
drugs_df = drugs_df[valid_drug_mask].copy()
drug_embeddings = drug_embeddings[valid_drug_mask.values]

print(f"Valid drugs: {len(drugs_df)}")

# ============================================================================
# Load SPRINT Target Embeddings
# ============================================================================
print("\n" + "=" * 80)
print("LOADING TARGET DATA")
print("=" * 80)

targets_df = pd.read_csv(os.path.join(SPRINT_DIR, 'targets.csv'))
target_embeddings = np.load(os.path.join(SPRINT_DIR, 'target_embeddings.npy'))

print(f"Loaded targets.csv: {len(targets_df)} targets")
print(f"Loaded target_embeddings.npy: shape {target_embeddings.shape}")

assert len(targets_df) == target_embeddings.shape[0], "Target count mismatch!"

print(f"Valid targets: {len(targets_df)}")

# ============================================================================
# Prepare Prediction DataFrames
# ============================================================================
print("\n" + "=" * 80)
print("PREPARING PREDICTION DATAFRAMES")
print("=" * 80)

embedding_dim = drug_embeddings.shape[1]
embedding_cols = [f'emb_{i}' for i in range(embedding_dim)]

# Drug predictions DataFrame
drug_pred_data = {'smiles': drugs_df['smiles_canonical'].values}
for i, col in enumerate(embedding_cols):
    drug_pred_data[col] = drug_embeddings[:, i]
drug_preds = pd.DataFrame(drug_pred_data)

print(f"\nDrug predictions:")
print(f"  Shape: {drug_preds.shape}")
print(f"  Unique SMILES: {drug_preds['smiles'].nunique()}")

# Target predictions DataFrame
target_pred_data = {'targetId': targets_df['target_id'].values}
for i, col in enumerate(embedding_cols):
    target_pred_data[col] = target_embeddings[:, i]
target_preds = pd.DataFrame(target_pred_data)

print(f"\nTarget predictions:")
print(f"  Shape: {target_preds.shape}")
print(f"  Unique targets: {target_preds['targetId'].nunique()}")

# ============================================================================
# Run DTR-Bench Evaluation with PCA Batch Correction Sweep
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING DTR-BENCH EVALUATION")
print("=" * 80)
print("\nTesting batch correction with 0-3 PCs removed (same as other baselines)...")

best_results = None
best_n_pcs = 0

for n_pcs in range(4):
    print(f"\n--- Testing with {n_pcs} PCs removed ---")
    
    if n_pcs == 0:
        # No PCA correction
        drug_preds_pca = drug_preds.copy()
        target_preds_pca = target_preds.copy()
    else:
        # Apply PCA batch correction (same method as other baselines)
        drug_ids = drug_preds['smiles'].copy()
        target_ids = target_preds['targetId'].copy()
        
        n_drugs = len(drug_preds)
        n_targets = len(target_preds)
        
        drug_feature_cols = [col for col in drug_preds.columns if col != 'smiles']
        target_feature_cols = [col for col in target_preds.columns if col != 'targetId']
        
        X_drug = drug_preds[drug_feature_cols].values
        X_target = target_preds[target_feature_cols].values
        
        # Combine, fit PCA, remove top PCs
        X = np.vstack([X_drug, X_target])
        
        pca = PCA(n_components=n_pcs)
        pca.fit(X)
        
        projection = pca.transform(X)
        reconstruction = projection @ pca.components_
        X_corrected = X - reconstruction
        
        # Separate back
        X_drug_corrected = X_corrected[:n_drugs]
        X_target_corrected = X_corrected[n_drugs:]
        
        # Rebuild DataFrames
        drug_preds_pca = pd.DataFrame(X_drug_corrected, columns=drug_feature_cols)
        drug_preds_pca.insert(0, 'smiles', drug_ids.values)
        
        target_preds_pca = pd.DataFrame(X_target_corrected, columns=target_feature_cols)
        target_preds_pca.insert(0, 'targetId', target_ids.values)
        
        print(f"  Removed variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Run evaluation
    results_tmp = submit_drug_target_mapping(
        drug_preds_pca,
        target_preds_pca,
    )
    
    # Track best by AUROC
    if best_results is None or results_tmp.get('auroc', 0) > best_results.get('auroc', 0):
        best_results = results_tmp
        best_n_pcs = n_pcs

print(f"\n✓ Best result with {best_n_pcs} PCs removed")

# ============================================================================
# Results Summary
# ============================================================================
print("\n" + "=" * 80)
print("SPRINT DTR-BENCH RESULTS SUMMARY")
print("=" * 80)
print(f"\n(Best result with {best_n_pcs} PCs removed for batch correction)")

k_list = [1, 5, 10, 50]

print(f"\nDrug -> Target Retrieval ({best_results.get('drug_to_target_queries', 0)} queries):")
print("{:<15} {:>12} {:>12} {:>12}".format('k', 'Precision@k', 'Hits@k', 'MRR@k'))
print("-" * 55)
for k in k_list:
    print("{:<15} {:>12.4f} {:>12.4f} {:>12.4f}".format(
        k,
        best_results.get(f'drug_to_target_precision@{k}', 0),
        best_results.get(f'drug_to_target_recall@{k}', 0),
        best_results.get(f'drug_to_target_mrr@{k}', 0)
    ))

print(f"\nTarget -> Drug Retrieval ({best_results.get('target_to_drug_queries', 0)} queries):")
print("{:<15} {:>12} {:>12} {:>12}".format('k', 'Precision@k', 'Hits@k', 'MRR@k'))
print("-" * 55)
for k in k_list:
    print("{:<15} {:>12.4f} {:>12.4f} {:>12.4f}".format(
        k,
        best_results.get(f'target_to_drug_precision@{k}', 0),
        best_results.get(f'target_to_drug_recall@{k}', 0),
        best_results.get(f'target_to_drug_mrr@{k}', 0)
    ))

print(f"\nGraph-Based Metrics:")
print(f"  AUROC: {best_results.get('auroc', 0):.4f}")
print(f"  AUPRC: {best_results.get('auprc', 0):.4f}")
print(f"  Positives: {best_results.get('n_positives', 0):,} / {best_results.get('n_total_pairs', 0):,}")

print("\n" + "=" * 80)
print("Evaluation complete!")
print("=" * 80)

