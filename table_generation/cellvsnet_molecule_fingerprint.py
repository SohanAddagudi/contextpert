import torch
import lightning as pl
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, ParameterGrid
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import warnings
import os
from pathlib import Path

from contextualized.regression.lightning_modules import ContextualizedCorrelation
from contextualized.data import CorrelationDataModule
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import EarlyStopping

seed_everything(10, workers=True)

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])

PATH_L1000 = DATA_DIR / 'pert_type_csvs' / f'trt_cp.csv'
PATH_CTLS = DATA_DIR / 'ctrls.csv'
PATH_SPLIT_MAP = DATA_DIR / 'gene_embeddings' / 'unseen_perturbation_splits' / 'trt_cp_split_map.csv'

OUTPUT_DIR = Path('./morgan_model_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving outputs to {OUTPUT_DIR.resolve()}")

N_DATA_PCS   = 50    
PERTURBATION_HOLDOUT_SIZE = 0.2  
RANDOM_STATE = 10
SUBSAMPLE_FRACTION = None  

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=1024)  

def smiles_to_morgan_fp(smiles, generator=morgan_gen):
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            warnings.warn(f"Invalid SMILES: {smiles}")
            return np.zeros(generator.GetOptions().fpSize)
        
        fp = generator.GetFingerprint(mol)
        return np.array(fp)
        
    except Exception as e:
        warnings.warn(f"Error processing SMILES {smiles}: {str(e)}")
        return np.zeros(generator.GetOptions().fpSize)

df = pd.read_csv(PATH_L1000, engine='pyarrow')

pert_to_fit_on = ['trt_cp']
df = df[df['pert_type'].isin(pert_to_fit_on)]

bad = (
    (df['distil_cc_q75'] < 0.2) | (df['distil_cc_q75'] == -666) | (df['distil_cc_q75'].isna()) |
    (df['pct_self_rank_q25'] > 5) | (df['pct_self_rank_q25'] == -666) | (df['pct_self_rank_q25'].isna())
)
df = df[~bad]

df = df.dropna(subset=['canonical_smiles'])
df = df[df['canonical_smiles'] != '']

print(f"Processing {len(df)} samples with valid SMILES...")

if SUBSAMPLE_FRACTION is not None:
    df = df.sample(frac=SUBSAMPLE_FRACTION, random_state=RANDOM_STATE)
    print(f"Subsampled to {len(df)} samples ({SUBSAMPLE_FRACTION*100}% of data)")

print(f"Loading train/test split from: {PATH_SPLIT_MAP}")
split_map_df = pd.read_csv(PATH_SPLIT_MAP)
split_map_df = split_map_df.rename(columns={'split': 'dataset_split'}) 

df = df.merge(split_map_df[['inst_id', 'dataset_split']], on='inst_id', how='inner')

if len(df) < len(split_map_df):
    print(f"Warning: {len(split_map_df) - len(df)} samples from the split map were not found in the main data after QC/filters.")

df_train = df[df['dataset_split'] == 'train'].copy()
df_test = df[df['dataset_split'] == 'test'].copy()

print(f"Sample split (from file): {len(df_train)} train, {len(df_test)} test samples")

smiles_train = df_train['canonical_smiles'].unique()
smiles_test = df_test['canonical_smiles'].unique()
unique_smiles = df['canonical_smiles'].unique()
print(f"Perturbation split (from file): {len(smiles_train)} train, {len(smiles_test)} test perturbations")

pert_time_mean = None
pert_dose_mean = None

for df_split, split_name in [(df_train, 'train'), (df_test, 'test')]:
    df_split['ignore_flag_pert_time'] = (df_split['pert_time'] == -666).astype(int)
    df_split['ignore_flag_pert_dose'] = (df_split['pert_dose'] == -666).astype(int)

    for col in ['pert_time', 'pert_dose']:
        if split_name == 'train':
            mean_val = df_split.loc[df_split[col] != -666, col].mean()
            if col == 'pert_time':
                pert_time_mean = mean_val
            else:
                pert_dose_mean = mean_val
        else:
            mean_val = pert_time_mean if col == 'pert_time' else pert_dose_mean
        
        df_split[col] = df_split[col].replace(-666, mean_val)

def process_data_split(df_split, split_name):
    numeric_cols   = df_split.select_dtypes(include=[np.number]).columns
    drop_cols      = ['pert_dose', 'pert_dose_unit', 'pert_time',
                      'distil_cc_q75', 'pct_self_rank_q25']
    feature_cols   = [c for c in numeric_cols if c not in drop_cols]
    X_raw          = df_split[feature_cols].values

    print(f"Generating Morgan fingerprints for {split_name} set...")
    morgan_fps = []
    for smiles in df_split['canonical_smiles']:
        fp = smiles_to_morgan_fp(smiles)
        morgan_fps.append(fp)

    morgan_fps = np.array(morgan_fps)
    print(f"Generated Morgan fingerprints for {split_name}: shape {morgan_fps.shape}")

    pert_unit_dummies  = pd.get_dummies(df_split['pert_dose_unit'], drop_first=True)

    pert_time    = df_split['pert_time'  ].to_numpy().reshape(-1, 1)
    pert_dose    = df_split['pert_dose'  ].to_numpy().reshape(-1, 1)
    ignore_time = df_split['ignore_flag_pert_time'].to_numpy().reshape(-1, 1)
    ignore_dose = df_split['ignore_flag_pert_dose'].to_numpy().reshape(-1, 1)

    return X_raw, morgan_fps, pert_unit_dummies, pert_time, pert_dose, ignore_time, ignore_dose

X_raw_train, morgan_fps_train, pert_unit_dummies_train, pert_time_train, pert_dose_train, ignore_time_train, ignore_dose_train = process_data_split(df_train, 'train')
X_raw_test, morgan_fps_test, pert_unit_dummies_test, pert_time_test, pert_dose_test, ignore_time_test, ignore_dose_test = process_data_split(df_test, 'test')

print("Applying improved scaling strategy...")

scaler_genes = StandardScaler()
X_train_scaled = scaler_genes.fit_transform(X_raw_train)
X_test_scaled = scaler_genes.transform(X_raw_test)
print(f"Gene expression scaled: train {X_train_scaled.shape}, test {X_test_scaled.shape}")

scaler_morgan = StandardScaler()
morgan_train_scaled = morgan_fps_train.astype(float)
morgan_test_scaled = morgan_fps_test.astype(float)
print(f"Morgan fingerprints scaled: train {morgan_train_scaled.shape}, test {morgan_test_scaled.shape}")

ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)          
unique_cells_train = np.sort(df_train['cell_id'].unique())
unique_cells_test = np.sort(df_test['cell_id'].unique())
unique_cells_all = np.sort(np.union1d(unique_cells_train, unique_cells_test))

ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells_all)]

scaler_ctrls = StandardScaler()
ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)

n_cells = ctrls_scaled.shape[0]
n_ctrl_pcs = min(50, n_cells)

pca_ctrls = PCA(n_components=n_ctrl_pcs, random_state=RANDOM_STATE)
ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)         

cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))

if not cell2vec:
    raise ValueError(
        "No common cell IDs found between lincs1000.csv and embeddings/ctrls.csv. "
        "Cannot proceed. Please check your data files."
    )

print(f"Loaded and processed control embeddings for {len(cell2vec)} unique cells.")
def build_context_matrix_improved(df_split, morgan_fps_scaled, pert_time, pert_dose, 
                                  ignore_time, ignore_dose, split_name, scaler_context=None, is_train=False):
    
    cell_ids = df_split['cell_id'].to_numpy()
    unique_cells_split = np.sort(df_split['cell_id'].unique())
    
    all_continuous_context = []
    valid_cells = []
    
    for cell_id in unique_cells_split:
        if cell_id not in cell2vec:
            print(f"Warning: Cell {cell_id} not found in control embeddings, skipping...")
            continue
            
        mask = cell_ids == cell_id
        if mask.sum() == 0:
            continue
            
        valid_cells.append(cell_id)
        
        C_continuous = np.hstack([
            np.tile(cell2vec[cell_id], (mask.sum(), 1)),  
            pert_time[mask],                              
            pert_dose[mask],                              
        ])
        all_continuous_context.append(C_continuous)
    
    if is_train:
        all_continuous_combined = np.vstack(all_continuous_context)
        scaler_context = StandardScaler()
        scaler_context.fit(all_continuous_combined)
        print(f"Fitted context scaler on {all_continuous_combined.shape} continuous context features")
    
    if scaler_context is None:
        raise ValueError("scaler_context must be provided for non-training data")
    
    X_lst, C_lst, cell_lst = [], [], []
    
    for i, cell_id in enumerate(valid_cells):
        mask = cell_ids == cell_id
        X_cell = X_train_scaled[mask] if split_name == 'train' else X_test_scaled[mask]
        
        C_continuous_scaled = scaler_context.transform(all_continuous_context[i])
        
        n_samples = mask.sum()
        
        C_cell = np.hstack([
            C_continuous_scaled,                         
            morgan_fps_scaled[mask],                        
            ignore_time[mask],                             
            ignore_dose[mask],
        ])

        X_lst.append(X_cell)
        C_lst.append(C_cell)
        cell_lst.append(cell_ids[mask])

    if not X_lst:
        raise RuntimeError(f"No data collected for {split_name} set.")
    
    X_final = np.vstack(X_lst)
    C_final = np.vstack(C_lst)
    cell_ids_final = np.concatenate(cell_lst)
    
    return X_final, C_final, cell_ids_final, scaler_context

print("Building context matrices with improved scaling...")

X_train, C_train, cell_ids_train, scaler_context = build_context_matrix_improved(
    df_train, morgan_train_scaled, pert_time_train, pert_dose_train,
    ignore_time_train, ignore_dose_train, 'train', is_train=True
)

X_test, C_test, cell_ids_test, _ = build_context_matrix_improved(
    df_test, morgan_test_scaled, pert_time_test, pert_dose_test,
    ignore_time_test, ignore_dose_test, 'test', scaler_context=scaler_context
)

print(f'Context matrix:   train {C_train.shape}   test {C_test.shape}')
print(f'Feature matrix:   train {X_train.shape}   test {X_test.shape}')

print("Applying PCA with improved scaling...")

pca_data = PCA(n_components=N_DATA_PCS, random_state=RANDOM_STATE)
X_train_pca = pca_data.fit_transform(X_train)
X_test_pca = pca_data.transform(X_test)

pca_scaler = StandardScaler()
X_train_norm = pca_scaler.fit_transform(X_train_pca)
X_test_norm = pca_scaler.transform(X_test_pca)

print(f'Normalized PCA features: train {X_train_norm.shape}   test {X_test_norm.shape}')

train_group_ids = cell_ids_train
test_group_ids = cell_ids_test
X_train = X_train_norm
X_test = X_test_norm
from contextualized.baselines.networks import CorrelationNetwork
pop_model = CorrelationNetwork()
pop_model.fit(X_train)
print(f"Train MSE: {pop_model.measure_mses(X_train).mean()}")
print(f"Test MSE: {pop_model.measure_mses(X_test).mean()}")

import wandb

contextualized_model = ContextualizedCorrelation(
    context_dim=C_train.shape[1],
    x_dim=X_train.shape[1],
    encoder_type='mlp',
    num_archetypes=30,
)

train_indices, val_indices = train_test_split(
    np.arange(len(X_train)), test_size=0.2, random_state=RANDOM_STATE
)
C_val = C_train[val_indices]
X_val = X_train[val_indices]
C_train_split = C_train[train_indices]
X_train_split = X_train[train_indices]

datamodule = CorrelationDataModule(
    C_train=C_train_split,
    X_train=X_train_split,
    C_val=C_val,
    X_val=X_val,
    C_test=C_test,
    X_test=X_test,
    C_predict=np.concatenate((C_train, C_test), axis=0),
    X_predict=np.concatenate((X_train, X_test), axis=0),
    batch_size=32,
)

checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
    dirpath=OUTPUT_DIR,
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    filename='best_model',
)
early_stop_callback = EarlyStopping(
   monitor='val_loss', 
   patience=5,         
   mode='min'          
)

trainer = Trainer(
    max_epochs=25,
    accelerator='auto',
    devices='auto',
    callbacks=[checkpoint_callback, early_stop_callback],
    deterministic=True,
)
trainer.fit(contextualized_model, datamodule=datamodule)

print(f"Testing model on training data...")
trainer.test(contextualized_model, datamodule.train_dataloader())
print(f"Testing model on test data...")
trainer.test(contextualized_model, datamodule.test_dataloader())
print(checkpoint_callback.best_model_path)
from contextualized.callbacks import PredictionWriter
from pathlib import Path

output_dir = Path(checkpoint_callback.best_model_path).parent / 'predictions'
writer_callback = PredictionWriter(
    output_dir=output_dir,
    write_interval='batch',
)
trainer = Trainer(
    accelerator='auto',
    devices='auto',
    callbacks=[checkpoint_callback, writer_callback],
    deterministic=True,
)
print("Making predictions on full dataset (train + test)...")
_ = trainer.predict(contextualized_model, datamodule=datamodule)

import torch
import glob

C_train_32 = C_train.astype(np.float32)
C_test_32 = C_test.astype(np.float32)
C_predict_32 = np.concatenate((C_train_32, C_test_32), axis=0)
C_predict_hashable = [tuple(row) for row in C_predict_32]

all_correlations = {}
all_betas = {}
all_mus = {}
pred_files = glob.glob(str(output_dir / 'predictions_*.pt'))
for file in pred_files:
    preds = torch.load(file)
    for context, correlation, beta, mu in zip(preds['contexts'], preds['correlations'], preds['betas'], preds['mus']):
        context_tuple = tuple(context.to(torch.float32).tolist())
        all_correlations[context_tuple] = correlation.cpu().numpy()
        all_betas[context_tuple] = beta.cpu().numpy()
        all_mus[context_tuple] = mu.cpu().numpy()

correlations_train = np.array([all_correlations[c] for c in [tuple(row) for row in C_train_32]])
correlations_test = np.array([all_correlations[c] for c in [tuple(row) for row in C_test_32]])
betas_train = np.array([all_betas[c] for c in [tuple(row) for row in C_train_32]])
betas_test = np.array([all_betas[c] for c in [tuple(row) for row in C_test_32]])
mus_train = np.array([all_mus[c] for c in [tuple(row) for row in C_train_32]])
mus_test = np.array([all_mus[c] for c in [tuple(row) for row in C_test_32]])

correlations_full = np.concatenate([correlations_train, correlations_test], axis=0)
betas_full = np.concatenate([betas_train, betas_test], axis=0)
mus_full = np.concatenate([mus_train, mus_test], axis=0)

X_full = np.concatenate([X_train, X_test], axis=0)
C_full = np.concatenate([C_train, C_test], axis=0)

train_indices_full = np.arange(len(X_train))
test_indices_full = np.arange(len(X_train), len(X_train) + len(X_test))

print(f"Full dataset predictions compiled:")
print(f"  Correlations shape: {correlations_full.shape}")
print(f"  Betas shape: {betas_full.shape}")
print(f"  Mus shape: {mus_full.shape}")
print(f"  Features shape: {X_full.shape}")
print(f"  Context shape: {C_full.shape}")

def measure_mses(betas, mus, X):
    mses = np.zeros(len(X))
    for i in range(len(X)):
        sample_mse = 0
        for j in range(X.shape[-1]):
            for k in range(X.shape[-1]):
                residual = X[i, j] - betas[i, j, k] * X[i, k] - mus[i, j, k]
                sample_mse += residual**2
        mses[i] = sample_mse / (X.shape[-1] ** 2)
    return mses

mse_train = measure_mses(betas_train, mus_train, X_train)
mse_test = measure_mses(betas_test, mus_test, X_test)
mse_full = measure_mses(betas_full, mus_full, X_full)

print(f"Train MSEs: {mse_train.mean()}")
print(f"Test MSEs: {mse_test.mean()}")
print(f"Full dataset MSEs: {mse_full.mean()}")

results_df = pd.DataFrame({
    'split': (['train'] * len(df_train)) + (['test'] * len(df_test)),
    'sample_idx': list(range(len(df_train))) + list(range(len(df_test))),
    'mse': mse_full,
})

df_full = pd.concat([df_train.reset_index(drop=True), df_test.reset_index(drop=True)], ignore_index=True)

results_df = pd.DataFrame({
    'split': df_full['dataset_split'].values, 
    'sample_idx': list(range(len(df_train))) + list(range(len(df_test))),
    'mse': mse_full,
    'cell_id': df_full['cell_id'].values,
    'canonical_smiles': df_full['canonical_smiles'].values,
    'pert_time': df_full['pert_time'].values,
    'pert_dose': df_full['pert_dose'].values,
    'inst_id': df_full['inst_id'].values 
})

print(f"\nResults dataframe shape: {results_df.shape}")
print(f"Results dataframe columns: {results_df.columns.tolist()}")

print(f"\nPer-cell performance breakdown:")
print("Cell ID           Train MSE    Test MSE      Train N  Test N")
print("─" * 60)

all_unique_cells = np.union1d(cell_ids_train, cell_ids_test)

for cell_id in sorted(all_unique_cells):
    tr_mask = cell_ids_train == cell_id
    te_mask = cell_ids_test == cell_id
    
    tr_mse = mse_train[tr_mask].mean() if tr_mask.any() else np.nan
    te_mse = mse_test[te_mask].mean() if te_mask.any() else np.nan
    tr_n = tr_mask.sum()
    te_n = te_mask.sum()
    
    if tr_n > 0 or te_n > 0:
        print(f'{cell_id:<15}  {tr_mse:8.4f}    {te_mse:8.4f}    {tr_n:6d}    {te_n:6d}')

print(f"\n" + "="*80)
print("PERTURBATION HOLDOUT SUMMARY:")
print(f"  Total unique SMILES: {len(unique_smiles)}")
print(f"  Training SMILES: {len(smiles_train)} ({len(smiles_train)/len(unique_smiles)*100:.1f}%)")
print(f"  Test SMILES: {len(smiles_test)} ({len(smiles_test)/len(unique_smiles)*100:.1f}%)")
print(f"  Training samples: {len(df_train)}")
print(f"  Test samples: {len(df_test)}")
print("="*80)

print(f"\nFULL DATASET INFERENCE SUMMARY:")
print(f"  Total samples with predictions: {len(results_df)}")
print(f"  Training samples: {(results_df['split'] == 'train').sum()}")
print(f"  Test samples: {(results_df['split'] == 'test').sum()}")
print(f"  Average MSE across full dataset: {results_df['mse'].mean():.4f}")
print("="*80)

print(f"\nSaving results to {OUTPUT_DIR.resolve()}:")
results_df.to_csv(OUTPUT_DIR / 'full_dataset_predictions.csv', index=False)
np.save(OUTPUT_DIR / 'full_dataset_correlations.npy', correlations_full)
np.save(OUTPUT_DIR / 'full_dataset_betas.npy', betas_full)
np.save(OUTPUT_DIR / 'full_dataset_mus.npy', mus_full)
