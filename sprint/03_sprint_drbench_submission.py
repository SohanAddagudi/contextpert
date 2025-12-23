import os
import pandas as pd
import numpy as np

from contextpert import submit_drug_disease_cohesion
from contextpert.utils import canonicalize_smiles

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']
SPRINT_DIR = os.path.join(DATA_DIR, 'sprint')

print("=" * 80)
print("SPRINT DR-BENCH EVALUATION")
print("=" * 80)
print("\nThis evaluates SPRINT drug embeddings on the DR-Bench")
print("(drug-disease cohesion) benchmark.\n")

# ============================================================================
# Load SPRINT drug embeddings
# ============================================================================
print("Loading SPRINT drug embeddings...")

# Load the drugs CSV to get SMILES order
drugs_df = pd.read_csv(os.path.join(SPRINT_DIR, 'drugs.csv'))
print(f"  Loaded drugs.csv: {len(drugs_df)} drugs")

# Load embeddings
embeddings_path = os.path.join(SPRINT_DIR, 'drug_embeddings.npy')
drug_embeddings = np.load(embeddings_path)
print(f"  Loaded embeddings: shape {drug_embeddings.shape}")

# Verify alignment
assert len(drugs_df) == drug_embeddings.shape[0], \
    f"Mismatch: {len(drugs_df)} drugs but {drug_embeddings.shape[0]} embeddings"

# ============================================================================
# Prepare prediction DataFrame
# ============================================================================
print("\nPreparing prediction DataFrame...")

# Create embedding column names
embedding_dim = drug_embeddings.shape[1]
embedding_cols = [f'emb_{i}' for i in range(embedding_dim)]

# Canonicalize SMILES for consistent matching
print("  Canonicalizing SMILES...")
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

# Filter out failed canonicalizations
valid_mask = drugs_df['smiles_canonical'].notna()
drugs_df = drugs_df[valid_mask].copy()
drug_embeddings = drug_embeddings[valid_mask.values]

print(f"  Valid drugs after canonicalization: {len(drugs_df)}")

# Build prediction DataFrame
pred_data = {'smiles': drugs_df['smiles_canonical'].values}
for i, col in enumerate(embedding_cols):
    pred_data[col] = drug_embeddings[:, i]

pred_df = pd.DataFrame(pred_data)

print(f"\nPrediction DataFrame:")
print(f"  Shape: {pred_df.shape}")
print(f"  Unique SMILES: {pred_df['smiles'].nunique()}")
print(f"  Embedding dimensions: {embedding_dim}")

# ============================================================================
# Run DR-Bench Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING DR-BENCH EVALUATION")
print("=" * 80)

results = submit_drug_disease_cohesion(pred_df, mode='lincs')

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SPRINT DR-BENCH RESULTS SUMMARY")
print("=" * 80)

k_list = [1, 5, 10, 25, 50]
print(f"\nEvaluated {results['n_queries']} queries across {results['n_unique_diseases']} diseases")

print("\n{:<20} {:>10} {:>15} {:>15} {:>10}".format(
    'k', 'Hits@k', 'Prec@k (micro)', 'Prec@k (macro)', 'MRR@k'))
print("-" * 70)

for k in k_list:
    print("{:<20} {:>10.4f} {:>15.4f} {:>15.4f} {:>10.4f}".format(
        k,
        results[f'hits@{k}'],
        results[f'precision@{k}_micro'],
        results[f'precision@{k}_macro'],
        results[f'mrr@{k}']
    ))

print("\n" + "=" * 80)
print("Evaluation complete!")
print("=" * 80)

