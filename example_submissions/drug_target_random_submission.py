import os
import pandas as pd
import numpy as np

from contextpert import submit_drug_target_mapping

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


print("="*80)
print("DRUG-TARGET DUMMY EVALUATION")
print("="*80)
print("\nThis example uses random vectors to test the drug-target mapping")
print("evaluation framework.\n")

# Load reference data to get list of drugs and targets
pairs_df = pd.read_csv(os.path.join(DATA_DIR, 'opentargets/drug_target_pairs_csv/drug_target_pairs.csv'))

drug_smiles_list = pairs_df['smiles'].unique().tolist()
target_id_list = pairs_df['targetId'].unique().tolist()

print(f"Reference data:")
print(f"  {len(drug_smiles_list)} unique drugs")
print(f"  {len(target_id_list)} unique targets")
print(f"  {len(pairs_df)} drug-target pairs")

# Create dummy drug representations: random 100-dimensional vectors
print("\nGenerating dummy drug representations...")
np.random.seed(42)
n_drugs = len(drug_smiles_list)
drug_repr_dim = 100

drug_pred_data = {'smiles': drug_smiles_list}
drug_random_vectors = np.random.randn(n_drugs, drug_repr_dim)
for i in range(drug_repr_dim):
    drug_pred_data[f'drug_dim_{i}'] = drug_random_vectors[:, i]

drug_preds = pd.DataFrame(drug_pred_data)

print(f"Created drug representations: {drug_preds.shape}")

# Create dummy target representations: random 100-dimensional vectors
print("\nGenerating dummy target representations...")
np.random.seed(123)  # Different seed for targets
n_targets = len(target_id_list)
target_repr_dim = 100

target_pred_data = {'targetId': target_id_list}
target_random_vectors = np.random.randn(n_targets, target_repr_dim)
for i in range(target_repr_dim):
    target_pred_data[f'target_dim_{i}'] = target_random_vectors[:, i]

target_preds = pd.DataFrame(target_pred_data)

print(f"Created target representations: {target_preds.shape}")

# Submit for evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION (LINCS MODE)")
print("="*80)
print("Using 'lincs' mode (default) to evaluate on LINCS-filtered subset")
print("This ensures drugs and targets are present in high-quality LINCS datasets\n")

results = submit_drug_target_mapping(drug_preds, target_preds, mode='lincs')

print("\nEvaluation complete! These are baseline results using random vectors.")
print("Expected: Random performance (AUROC ~0.5, low retrieval metrics)")
print("\nNote: Use mode='full' to evaluate on all OpenTargets drug-target pairs")
