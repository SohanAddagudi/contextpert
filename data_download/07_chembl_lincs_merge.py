import os
import pandas as pd

data_dir = os.environ['CONTEXTPERT_DATA_DIR']

file1 = pd.read_csv(os.path.join(data_dir, "opentargets", "cancer_target_drug_phase4_csv", "part-00000-3b99aee4-2b44-46d7-9822-63f7222a3695-c000.csv"))  # Processed OpenTargets file path
file2 = pd.read_csv(os.path.join(data_dir, "full_lincs_with_chembl.csv"))  # Path to LINCS file with 'chembl_id', generated from brd_to_chembl.py

merged_df = pd.merge(file2, file1, left_on="chembl_id", right_on="drugId", how="left")
merged_df.to_csv(os.path.join(data_dir, "full_lincs_with_disease.csv"), index=False)
