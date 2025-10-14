import pandas as pd
import mygene
import io

# 1. Load drug target pairs
df = pd.read_csv('/home/user/screening2/contextpert/cp_gene/drug_target_pairs.csv')

# 2. Get unique Ensembl IDs to query
ensg_ids = df['targetId'].unique().tolist()

# 3. Initialize mygene client and perform the query
mg = mygene.MyGeneInfo()
gene_info = mg.querymany(ensg_ids, scopes='ensembl.gene', fields='symbol', species='human')

# 4. Create a mapping from Ensembl ID to gene symbol
id_to_symbol_map = {info['query']: info.get('symbol') for info in gene_info if 'symbol' in info}

# 5. Create the new 'symbol' column using the map
df['symbol'] = df['targetId'].map(id_to_symbol_map)

# 6. Drop the old 'targetId' column and reorder
df = df.drop('targetId', axis=1)
cols = ['drugId', 'symbol', 'smiles', 'prefName']
symbols_df = df[cols]

# Display the symbols DataFrame
print("--- Symbols DataFrame ---")
print(symbols_df)
print("-" * 25)

# 7. Load pertinfo
pertinfo = pd.read_csv('/home/user/contextulized/GSE92742_Broad_LINCS_pert_info.txt', sep='\t')

# Filter the DataFrame by the specified 'pert_type' values
filtered_df = pertinfo[pertinfo['pert_type'].isin(['trt_oe', 'trt_lig', 'trt_sh', 'trt_cp'])]

# Select the desired columns
final_df = filtered_df[['pert_id', 'pert_iname', 'pert_type']].copy()

# 8. Load inst_chembl
inst_chembl = pd.read_csv('/home/user/screening2/contextpert/cp_gene/inst_chembl.csv')
print("\n--- Inst_chembl DataFrame ---")
print(inst_chembl)
print("-" * 29)

# 9. For trt_cp rows, map pert_id -> chembl_id -> symbol
symbols_df_unique = symbols_df.drop_duplicates(subset='drugId', keep='first')

# Create mapping: pert_id -> chembl_id
pert_to_chembl = dict(zip(inst_chembl['pert_id'], inst_chembl['chembl_id']))

# Create mapping: chembl_id (drugId) -> symbol
chembl_to_symbol = dict(zip(symbols_df_unique['drugId'], symbols_df_unique['symbol']))

# Isolate trt_cp rows to work on them
trt_cp_mask = final_df['pert_type'] == 'trt_cp'
initial_cp_count = trt_cp_mask.sum()

# For trt_cp rows, replace pert_iname with the mapped symbol
final_df.loc[trt_cp_mask, 'pert_iname'] = (
    final_df.loc[trt_cp_mask, 'pert_id']
    .map(pert_to_chembl)  # pert_id -> chembl_id
    .map(chembl_to_symbol)  # chembl_id -> symbol
)

final_df.dropna(subset=['pert_iname'], inplace=True)

# Calculate final stats
final_cp_count = (final_df['pert_type'] == 'trt_cp').sum()
removed_count = initial_cp_count - final_cp_count

print("\n--- trt_cp Statistics ---")
print(f"Initial 'trt_cp' count: {initial_cp_count}")
print(f"Removed (no symbol found): {removed_count}")
print(f"Final 'trt_cp' count: {final_cp_count}")

print("\nFinal DataFrame:")
print(final_df)

final_df.to_csv('pert_target.csv', index=False)
