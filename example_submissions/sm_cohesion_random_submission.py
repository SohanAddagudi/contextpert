import os
import pandas as pd
import numpy as np

from contextpert.evaluate import submit_drug_disease_cohesion

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


# Load reference data to get list of drugs
disease_drug_df = pd.read_csv(os.path.join(DATA_DIR, 'opentargets/disease_drug_triples_csv/disease_drug_triples.csv'))
drug_smiles_list = disease_drug_df['smiles'].unique().tolist()

# Create dummy predictions: random 100-dimensional vectors
np.random.seed(0)
n_drugs = len(drug_smiles_list)
repr_dim = 100
pred_data = {'smiles': drug_smiles_list}
random_vectors = np.random.randn(n_drugs, repr_dim)
for i in range(repr_dim):
    pred_data[f'dim_{i}'] = random_vectors[:, i]
my_preds = pd.DataFrame(pred_data)

# Submit results
results = submit_drug_disease_cohesion(my_preds, mode='lincs')
