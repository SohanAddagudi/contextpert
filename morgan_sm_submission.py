import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from evaluate import submit_sm_disease_cohesion

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


print("="*80)
print("MORGAN FINGERPRINT EVALUATION")
print("="*80)
print("\nThis example uses Morgan fingerprints (ECFP4, radius=2, 2048 bits)")
print("as molecular representations for the evaluation framework.\n")

# Load reference data to get list of drugs
disease_drug_df = pd.read_csv(os.path.join(DATA_DIR, 'opentargets/disease_drug_triples_csv/disease_drug_triples.csv'))
print(disease_drug_df.head())

# Get unique drugs (already have SMILES in this dataset)
drug_smiles_list = disease_drug_df['smiles'].unique().tolist()

# Generate Morgan fingerprints for all molecules
print(f"\nGenerating Morgan fingerprints for {len(drug_smiles_list)} drugs...")
radius = 2  # radius=2 corresponds to ECFP4
n_bits = 2048  # standard fingerprint size

# Create Morgan fingerprint generator using the new API
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

morgan_fps = []
valid_smiles = []

for smiles in tqdm(drug_smiles_list, desc="Generating fingerprints"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Generate Morgan fingerprint as bit vector using new API
        fp = morgan_gen.GetFingerprint(mol)
        # Convert to numpy array
        fp_array = np.zeros(n_bits, dtype=np.float32)
        for i in range(n_bits):
            fp_array[i] = fp[i]
        morgan_fps.append(fp_array)
        valid_smiles.append(smiles)
    else:
        print(f"Warning: Could not parse SMILES: {smiles}")

# Create prediction dataframe
pred_data = {'smiles': valid_smiles}
morgan_fps_array = np.array(morgan_fps)

for i in range(n_bits):
    pred_data[f'dim_{i}'] = morgan_fps_array[:, i]

my_preds = pd.DataFrame(pred_data)

print(f"\nCreated Morgan fingerprint representations for {len(my_preds)} drugs")
print(f"Fingerprint parameters: radius={radius}, n_bits={n_bits}")
print(f"Prediction dataframe shape: {my_preds.shape}")
print(f"\nFirst few rows of prediction dataframe:")
print(my_preds.iloc[:3, :5])  # Show first 3 rows, first 5 columns

# Submit for evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION")
print("="*80)

results = submit_sm_disease_cohesion(my_preds)

print("\nEvaluation complete! These are results using Morgan fingerprints (ECFP4).")
