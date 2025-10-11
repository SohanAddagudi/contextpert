import os
import pandas as pd
import numpy as np

from evaluate import submit_sm_disease_cohesion

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


print("="*80)
print("DUMMY EVALUATION")
print("="*80)
print("\nThis example demonstrates how to use the evaluation framework with random")
print("100-dimensional vectors. In practice, you would replace these with your")
print("learned representations (e.g., from a trained model).\n")

# Load reference data to get list of drugs
disease_drug_df = pd.read_csv(os.path.join(DATA_DIR, 'opentargets/disease_drug_triples_csv/disease_drug_triples.csv'))
print(disease_drug_df.head())

# Get unique drugs (already have SMILES in this dataset)
drug_smiles_list = disease_drug_df['smiles'].unique().tolist()

# Create dummy predictions: random 100-dimensional vectors
np.random.seed(42)
n_drugs = len(drug_smiles_list)
repr_dim = 100
pred_data = {'smiles': drug_smiles_list}
random_vectors = np.random.randn(n_drugs, repr_dim)
for i in range(repr_dim):
    pred_data[f'dim_{i}'] = random_vectors[:, i]
my_preds = pd.DataFrame(pred_data)

print(f"\nCreated dummy predictions for {len(my_preds)} drugs with {repr_dim}-dimensional vectors")
print(f"Prediction dataframe shape: {my_preds.shape}")
print(f"\nFirst few rows of prediction dataframe:")
print(my_preds.iloc[:3, :5])  # Show first 3 rows, first 5 columns

# Submit for evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION")
print("="*80)

results = submit_sm_disease_cohesion(my_preds)

print("\nEvaluation complete! These are baseline results using random vectors.")