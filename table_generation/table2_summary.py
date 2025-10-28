#!/usr/bin/env python
"""
Generate Table 2: Drug-Target Dataset Summary

Analyzes the LINCS-filtered drug-target pairs dataset and generates a summary
table showing:
- Total pairs
- Unique drugs
- Unique targets
- Average drugs per target
- Average targets per drug

Output: LaTeX table format
"""

import os
import pandas as pd

DATA_DIR = os.getenv('CONTEXTPERT_DATA_DIR')

print("=" * 80)
print("TABLE 2: DRUG-TARGET DATASET SUMMARY")
print("=" * 80)

# Load drug-target pairs
pairs_path = os.path.join(DATA_DIR, 'opentargets/drug_target_pairs_csv/drug_target_pairs_lincs.csv')
print(f"\nLoading drug-target pairs from: {pairs_path}")
pairs_df = pd.read_csv(pairs_path)

print(f"\nLoaded {len(pairs_df):,} drug-target pairs")

# Calculate summary statistics
total_pairs = len(pairs_df)
unique_drugs = pairs_df['smiles'].nunique()
unique_targets = pairs_df['targetId'].nunique()

# Calculate average drugs per target
drugs_per_target = pairs_df.groupby('targetId')['smiles'].nunique()
avg_drugs_per_target = drugs_per_target.mean()
std_drugs_per_target = drugs_per_target.std()

# Calculate average targets per drug
targets_per_drug = pairs_df.groupby('smiles')['targetId'].nunique()
avg_targets_per_drug = targets_per_drug.mean()
std_targets_per_drug = targets_per_drug.std()

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"\nTotal drug-target pairs:      {total_pairs:,}")
print(f"Unique drugs (SMILES):        {unique_drugs:,}")
print(f"Unique targets (Ensembl IDs): {unique_targets:,}")
print(f"Average drugs per target:     {avg_drugs_per_target:.2f} ± {std_drugs_per_target:.2f}")
print(f"Average targets per drug:     {avg_targets_per_drug:.2f} ± {std_targets_per_drug:.2f}")

# Generate LaTeX table
print("\n" + "=" * 80)
print("LATEX TABLE")
print("=" * 80)
print()

latex_table = r"""
\begin{table}[h]
\centering
\caption{Summary statistics for the LINCS-filtered drug-target interaction dataset.}
\label{tab:drug_target_summary}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total drug-target pairs & """ + f"{total_pairs:,}" + r""" \\
Unique drugs & """ + f"{unique_drugs:,}" + r""" \\
Unique targets & """ + f"{unique_targets:,}" + r""" \\
Avg. drugs per target & """ + f"{avg_drugs_per_target:.2f} $\\pm$ {std_drugs_per_target:.2f}" + r""" \\
Avg. targets per drug & """ + f"{avg_targets_per_drug:.2f} $\\pm$ {std_targets_per_drug:.2f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
""".strip()

print(latex_table)

# Save to file
output_path = os.path.join(os.path.dirname(__file__), '..', 'paper_tables_from_eval', 'table2_drug_target_summary.tex')
print(f"\n\nSaving LaTeX table to: {output_path}")
with open(output_path, 'w') as f:
    f.write(latex_table)

print("\n✓ Done")
