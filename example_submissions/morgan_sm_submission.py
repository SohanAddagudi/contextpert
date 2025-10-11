import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from contextpert import submit_sm_disease_cohesion

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


# Load reference data to get list of drugs
disease_drug_df = pd.read_csv(os.path.join(DATA_DIR, 'opentargets/disease_drug_triples_csv/disease_drug_triples.csv'))
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

# Submit
results = submit_sm_disease_cohesion(my_preds)
